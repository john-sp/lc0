/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <new>
#include <queue>
#include <thread>

#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/shared_params.h"
#include "utils/atomic_vector.h"
#include "utils/mutex.h"
#include "utils/numa.h"
#include "utils/trace.h"

namespace lczero {
namespace {

using Clock = std::chrono::high_resolution_clock;
#if __cpp_lib_hardware_interference_size >= 201703L
constexpr size_t kCacheLineSize = std::hardware_destructive_interference_size;
#else
constexpr size_t kCacheLineSize = 64;
#endif

class DemuxingComputation;

struct DemuxingWork {
  DemuxingComputation* source_ = nullptr;
  std::unique_ptr<NetworkComputation> computation_;
  int start_ = 0;
  int end_ : 31 = 0;
  bool first_in_queue_ : 1 = false;
  std::tuple<uint32_t, uint32_t> predicted_times_;

  DemuxingWork(int sample) : end_(sample) {}
  DemuxingWork(DemuxingComputation* source, int start, int end)
      : source_(source), start_(start), end_(end) {
    assert(start_ != end_);
  }

  void ProcessResults();

  auto operator<=>(const DemuxingWork& b) const { return end_ <=> b.end_; }
};

class DemuxingComputation;
class DemuxingChildBackend;

class DemuxingChildBackend {
 public:
  using AssingType = std::tuple<BackendAttributes, const NetworkCapabilities&>;
  using AssignFuture = std::future<AssingType>;
  using AssignPromise = std::promise<AssingType>;

  ~DemuxingChildBackend();

  AssignFuture Assign(const std::string& name,
                      const std::optional<WeightsFile>& weights,
                      const OptionsDict& opts, std::atomic<bool>& abort,
                      size_t shared_count, size_t shared_stride) {
    int numa_node = opts.GetOrDefault<int>("numa_node", -1);
    AssignPromise promise;
    auto rv = promise.get_future();
    threads_.emplace_back([this, name, &weights, opts,
                           promise = std::move(promise), &abort, numa_node,
                           shared_count, shared_stride]() mutable {
      try {
        if (numa_node >= 0) {
          Numa::BindThreadToNode(numa_node);
        }
        AssingType result =
            shared_count > 0 ? ConstructBackend(name, weights, opts)
                             : UseSharedBackend(this[-shared_stride].network_);
        std::vector<AssignFuture> other_backends;
        if (shared_count > 1) {
          other_backends.reserve(shared_count - 1);
          for (size_t i = 1; i < shared_count; i++) {
            other_backends.emplace_back(this[shared_stride * i].Assign(
                name, weights, opts, abort, 0, shared_stride * i));
          }
        }
        int nn_threads = opts.GetOrDefault<int>("demux_threads", 0);
        if (nn_threads == 0) {
          nn_threads = network_->GetThreads() + !network_->IsCpu();
        }
        for (int i = 1; i < nn_threads; i++) {
          threads_.emplace_back([&] {
            if (numa_node >= 0) {
              Numa::BindThreadToNode(numa_node);
            }
            Worker(abort);
          });
        }
        for (auto& f : other_backends) {
          // Wait for the other shared backends and check for exceptions.
          f.get();
        }
        promise.set_value(std::move(result));
      } catch (...) {
        promise.set_exception(std::current_exception());
        return;
      }
      Worker(abort);
    });
    return rv;
  }

  AssingType UseSharedBackend(std::shared_ptr<Network> shared) {
    network_ = shared;
    BackendAttributes attrs(*network_);
    batch_times_ns_ =
        std::make_unique<std::atomic<uint32_t>[]>(attrs.maximum_batch_size + 1);
    IdlePrediction prediction(Clock::now().time_since_epoch().count(), 0);
    idle_prediction_.store(prediction, std::memory_order_relaxed);
    return {attrs, network_->GetCapabilities()};
  }

  AssingType ConstructBackend(const std::string& name,
                              const std::optional<WeightsFile>& weights,
                              const OptionsDict& opts) {
    network_ = NetworkFactory::Get()->Create(name, weights, opts);
    BackendAttributes attrs(*network_);
    batch_times_ns_ =
        std::make_unique<std::atomic<uint32_t>[]>(attrs.maximum_batch_size + 1);
    IdlePrediction prediction(Clock::now().time_since_epoch().count(), 0);
    idle_prediction_.store(prediction, std::memory_order_relaxed);
    return {attrs, network_->GetCapabilities()};
  }

  int32_t GetIdlePrediction(Clock::time_point now) const {
    IdlePrediction prediction =
        idle_prediction_.load(std::memory_order_relaxed);
    if (prediction.queued_work_ns_ == 0) {
      return 0;
    }
    int64_t idle_time_ns =
        prediction.last_batch_completed_ - now.time_since_epoch().count();
    int64_t queued_work_ns = prediction.queued_work_ns_;
    return std::max<int64_t>(0, idle_time_ns + queued_work_ns);
  }

  std::tuple<uint32_t, uint32_t> AddBackendWork(int batch_size) {
    uint32_t batch_time = 0, original_batch_time = 0;
    original_batch_time = batch_time =
        batch_times_ns_[batch_size].load(std::memory_order_relaxed);

    if (batch_time == 0) {
      batch_time =
          batch_times_ns_[0].load(std::memory_order_relaxed) * batch_size;
    }
    auto idle = idle_prediction_.load(std::memory_order_relaxed);
    IdlePrediction new_idle = idle;
    do {
      if (idle.queued_work_ns_ == 0) {
        // Idle backend requires setting completion time to be when GPU gets the
        // work. We use CPU time to approximate the timing.
        auto now = Clock::now();
        new_idle.last_batch_completed_ = now.time_since_epoch().count();
      }
      new_idle.queued_work_ns_ += batch_time;
    } while (!idle_prediction_.compare_exchange_weak(
        idle, new_idle, std::memory_order_relaxed));
    return {batch_time, original_batch_time};
  }

  constexpr static int BatchTimeUpdateWeight = 32;

  void CompleteBackendWork(int batch_size,
                           std::tuple<uint32_t, uint32_t> predicted_times) {
    auto [predicted_batch_time, original_batch_time] = predicted_times;
    auto now = Clock::now();
    auto idle = idle_prediction_.load(std::memory_order_relaxed);
    IdlePrediction new_idle = idle;
    do {
      new_idle.last_batch_completed_ = now.time_since_epoch().count();
      new_idle.queued_work_ns_ -= predicted_batch_time;
    } while (!idle_prediction_.compare_exchange_weak(
        idle, new_idle, std::memory_order_relaxed));

    int32_t batch_time =
        now.time_since_epoch().count() - idle.last_batch_completed_;

    int32_t average_per_position = batch_time / batch_size;
    if (batch_time < 0) {
      // Protect against out of order clocks.
      return;
    }
    if (original_batch_time == 0) {
      if (batch_times_ns_[batch_size].compare_exchange_strong(
              original_batch_time, batch_time, std::memory_order_relaxed)) {
        // There was no average yet so we updated it to match the first batch
        // time.
        return;
      }
      // Other thread updated the prediction. We update the average using the
      // new prediction.
    }
    int32_t batch_time_delta =
        static_cast<int32_t>(batch_time - original_batch_time) /
        BatchTimeUpdateWeight;
    if (batch_time_delta == 0) {
      // No need to update atomic value if prediction was accurate.
      return;
    }
    batch_times_ns_[batch_size].fetch_add(batch_time_delta,
                                          std::memory_order_relaxed);

    uint32_t average = batch_times_ns_[0].load(std::memory_order_relaxed);
    if (average == 0) {
      if (batch_times_ns_[0].compare_exchange_strong(
              average, average_per_position, std::memory_order_relaxed)) {
        // There was no average yet so we updated it to match the first batch
        // time.
        return;
      }
    }
    int32_t average_delta =
        static_cast<int32_t>(average_per_position - average) /
        BatchTimeUpdateWeight;
    batch_times_ns_[0].fetch_add(average_delta, std::memory_order_relaxed);
  }

  void Enqueue(DemuxingWork* work) {
    {
      Mutex::Lock lock(mutex_);
      queue_.push(work);
    }
    dataready_cv_.notify_one();
  }

  void Abort() {
    {
      Mutex::Lock lock(mutex_);
    }
    dataready_cv_.notify_all();
  }

  void Worker(std::atomic<bool>& abort);

 private:
  struct IdlePrediction {
    uint64_t last_batch_completed_;
    uint64_t queued_work_ns_;
  };

  // Runtime constant variables
  std::vector<std::thread> threads_;
  std::shared_ptr<Network> network_;
  std::unique_ptr<std::atomic<uint32_t>[]> batch_times_ns_;
  // Atomically updated variables
  alignas(kCacheLineSize) std::atomic<IdlePrediction> idle_prediction_;
  // Mutex protected queue
  Mutex mutex_;
  std::condition_variable dataready_cv_;
  std::queue<DemuxingWork*> queue_ GUARDED_BY(mutex_);
};

class DemuxingBackend final : public Backend {
 public:
  DemuxingBackend(const std::optional<WeightsFile>& weights,
                  const OptionsDict& options,
                  const OptionsDict& backend_options)
      : backends_(std::max(size_t(1), backend_options.ListSubdicts().size() *
                                          backend_options.GetOrDefault<int>(
                                              "shared_backend_threads", 1))),
        backend_opts_(
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId)),
        weights_path_(
            options.Get<std::string>(SharedBackendParams::kWeightsId)) {
    int shared_threads =
        backend_options.GetOrDefault<int>("shared_backend_threads", 1);
    minimum_batch_step_ =
        backend_options.GetOrDefault<int>("min_batch_step", 1);
    UpdateConfiguration(options);
    const auto parents = backend_options.ListSubdicts();
    std::vector<DemuxingChildBackend::AssignFuture> capabilities;
    capabilities.reserve(std::max<size_t>(1, parents.size()));
    if (parents.empty()) {
      // If options are empty, or multiplexer configured in root object,
      // initialize on root object and default backend.
      auto backends = NetworkFactory::Get()->GetBackendsList();
      capabilities.emplace_back(
          AddBackend(0, backends[0], weights, backend_options, shared_threads));
    }

    int i = 0;
    for (const auto& name : parents) {
      capabilities.emplace_back(AddBackend(i++, name, weights,
                                           backend_options.GetSubdict(name),
                                           shared_threads));
    }
    i = 0;
    for (auto& future : capabilities) {
      assert(future.valid());
      auto [attr, caps] = future.get();
      if (i == 0) {
        i = 1;
        attrs_ = attr;
        input_format_ = caps.input_format;
      } else {
        attrs_ += attr;
        if (input_format_ != caps.input_format) {
          throw Exception("Incompatible input formats, " +
                          std::to_string(input_format_) + " vs " +
                          std::to_string(caps.input_format));
        }
      }
      if (attr.runs_on_cpu) {
        uses_cpu_backend_ = true;
      }
    }
    attrs_.maximum_batch_size =
        std::max(attrs_.recommended_batch_size, attrs_.maximum_batch_size);
    attrs_.maximum_batch_size = backend_options.GetOrDefault<int>(
        "max_batch", attrs_.maximum_batch_size);
    attrs_.recommended_batch_size =
        std::min(attrs_.maximum_batch_size, attrs_.recommended_batch_size);

    minimum_batch_step_ =
        std::clamp(minimum_batch_step_, 1, attrs_.recommended_batch_size);

    CERR << "Demuxing backend initialized with " << backends_.size()
         << " backends, maximum batch size: " << attrs_.maximum_batch_size
         << ", recommended batch size: " << attrs_.recommended_batch_size
         << ", preferred batch step: " << attrs_.preferred_batch_step
         << ", minimum batch step: " << minimum_batch_step_;
  }

  DemuxingChildBackend::AssignFuture AddBackend(
      int index, const std::string& name,
      const std::optional<WeightsFile>& weights, const OptionsDict& opts,
      int shared_threads) {
    const std::string backend = opts.GetOrDefault<std::string>("backend", name);

    return backends_[index].Assign(backend, weights, opts, abort_,
                                   shared_threads,
                                   backends_.size() / shared_threads);
  }

  std::unique_ptr<BackendComputation> CreateComputation() override;

  BackendAttributes GetAttributes() const override { return attrs_; }

  ~DemuxingBackend() { Abort(); }

  void Abort() {
    abort_.store(true, std::memory_order_relaxed);
    for (auto& b : backends_) {
      b.Abort();
    }
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    auto rv = Backend::UpdateConfiguration(options);
    if (rv != UPDATE_OK) return rv;
    if (backend_opts_ !=
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId)) {
      return NEED_RESTART;
    }
    if (weights_path_ !=
        options.Get<std::string>(SharedBackendParams::kWeightsId)) {
      return NEED_RESTART;
    }
    softmax_policy_temperature_ =
        1.0f / options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
    fill_empty_history_ = EncodeHistoryFill(
        options.Get<std::string>(SharedBackendParams::kHistoryFill));
    return UPDATE_OK;
  }

 private:
  std::vector<DemuxingChildBackend> backends_;
  BackendAttributes attrs_;
  pblczero::NetworkFormat::InputFormat input_format_;
  float softmax_policy_temperature_;
  FillEmptyHistory fill_empty_history_;
  int minimum_batch_step_ = 1;
  bool uses_cpu_backend_ = false;

  alignas(kCacheLineSize) std::atomic<int64_t> start_index_ = 0;
  alignas(kCacheLineSize) std::atomic<bool> abort_ = false;

  // Cache cold variables
  const std::string backend_opts_;
  const std::string weights_path_;

  friend class DemuxingComputation;
};

class DemuxingComputation final : public BackendComputation {
  std::tuple<const std::unique_ptr<NetworkComputation>&, int> GetParent(
      int sample) const {
    auto iter =
        std::lower_bound(children_.begin(), children_.end(), sample + 1);
    assert(iter != children_.end());
    assert(sample >= iter->start_);
    assert(sample < iter->end_);
    return {iter->computation_, sample - iter->start_};
  }

 public:
  DemuxingComputation(DemuxingBackend* backend)
      : backend_(backend), entries_(backend_->attrs_.maximum_batch_size) {}
  ~DemuxingComputation() {}

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    int transform;
    const size_t idx = entries_.emplace_back(NetworkComputationRequest{
        .input = EncodePositionForNN(backend_->input_format_, pos.pos, 8,
                                     backend_->fill_empty_history_, &transform),
        .legal_moves = MoveList(pos.legal_moves.begin(), pos.legal_moves.end()),
        .result = result,
        .transform = 0});
    entries_[idx].transform = transform;
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking(ComputationCallback callback) override;

  size_t UsedBatchSize() const override { return entries_.size(); }

  void NotifyFirstDone() {
    callback_(ComputationEvent::FIRST_BACKEND_IDLE);
    first_done_.notify_one();
  }

  void NotifyComplete() {
    int outstanding = dataready_.fetch_sub(1, std::memory_order_acq_rel);
    if (backend_->uses_cpu_backend_ && outstanding == 1) {
      last_done_.store(1, std::memory_order_release);
      last_done_.notify_one();
    }
  }

  void ProcessResults(const DemuxingWork& work);

 private:
  DemuxingBackend* backend_;
  AtomicVector<NetworkComputationRequest> entries_;
  std::vector<DemuxingWork> children_;
  ComputationCallback callback_;

  alignas(kCacheLineSize) std::atomic<int> dataready_ = 0;
  alignas(kCacheLineSize) std::atomic<int> first_done_ = 0;
  alignas(kCacheLineSize) std::atomic<int> last_done_ = 0;

  friend class DemuxingChildBackend;
};

void DemuxingWork::ProcessResults() { source_->ProcessResults(*this); }

void DemuxingComputation::ProcessResults(const DemuxingWork& work) {
  size_t size = work.end_ - work.start_;
  for (size_t i = 0; i < size; ++i) {
    entries_[work.start_ + i].ProcessResult(
        *work.computation_, i, backend_->softmax_policy_temperature_);
  }
}

std::unique_ptr<BackendComputation> DemuxingBackend::CreateComputation() {
  return std::make_unique<DemuxingComputation>(this);
}

DemuxingChildBackend::~DemuxingChildBackend() {
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
  while (!queue_.empty()) {
    queue_.front()->source_->NotifyComplete();
    queue_.pop();
  }
}

void DemuxingChildBackend::Worker(std::atomic<bool>& abort) {
  while (!abort.load(std::memory_order_relaxed)) {
    DemuxingWork* work = nullptr;
    {
      Mutex::Lock lock(mutex_);
      dataready_cv_.wait(lock.get_raw(), [&] {
        return abort.load(std::memory_order_relaxed) || !queue_.empty();
      });
      if (abort.load(std::memory_order_relaxed)) return;
      if (!queue_.empty()) {
        work = queue_.front();
        queue_.pop();
      }
    }
    if (work) {
      LCTRACE_FUNCTION_SCOPE;
      work->computation_ = network_->NewComputation();
      auto& entries = work->source_->entries_;
      for (int i = work->start_; i < work->end_; i++) {
        work->computation_->AddInput(std::move(entries[i].input));
      }
      auto batch_time = AddBackendWork(work->end_ - work->start_);
      work->computation_->ComputeBlocking();
      // TODO: This should read the time from the backend which could use more
      // accurate GPU timers. CPU time has potential for random extra delay
      // sometimes.
      CompleteBackendWork(work->end_ - work->start_, batch_time);
      int expected = 0;
      if (work->source_->first_done_.compare_exchange_strong(
              expected, 1, std::memory_order_relaxed)) {
        work->source_->NotifyFirstDone();
      }
      work->ProcessResults();
      work->source_->NotifyComplete();
    }
  }
}

void DemuxingComputation::ComputeBlocking(ComputationCallback callback) {
  LCTRACE_FUNCTION_SCOPE;
  assert(UsedBatchSize() != 0);
  callback_ = callback;
  // Calculate batch_step_ size split count.
  int step = backend_->attrs_.preferred_batch_step;
  if (UsedBatchSize() <
          backend_->minimum_batch_step_ * backend_->backends_.size() &&
      UsedBatchSize() >=
          backend_->minimum_batch_step_ * backend_->backends_.size() / 2) {
    step = backend_->minimum_batch_step_;
  }
  int splits = 1 + (UsedBatchSize() - 1) / step;
  int last_split_size = (UsedBatchSize() - 1) % step + 1;
  bool are_all_splits_full = last_split_size == step;
  // Calculate the minimum number of splits per backend.
  int split_size_per_backend = splits / backend_->backends_.size();
  // Calculate how many backends get extra work.
  int extra_split_backends =
      splits - split_size_per_backend * backend_->backends_.size();

  struct BackendSortingOrder {
    uint32_t idx;
    uint32_t work_left;
    auto operator<=>(const BackendSortingOrder& b) const {
      return work_left <=> b.work_left;
    }
  };

  // Do basic load balancing based on predicted time until idle.
  std::vector<BackendSortingOrder> backend_order;
  backend_order.reserve(backend_->backends_.size());

  auto now = Clock::now();
  for (uint32_t idx = 0; idx < backend_->backends_.size(); idx++) {
    const auto& b = backend_->backends_[idx];
    int32_t idle_time = b.GetIdlePrediction(now);
    backend_order.emplace_back(idx, static_cast<uint32_t>(idle_time));
  }

  std::sort(backend_order.begin(), backend_order.end());

  // Find the first backend which got less work from the previous batch.
  size_t start_index = 0;
  size_t end_index = extra_split_backends;
  size_t work_start = 0;
  int work_items = split_size_per_backend > 0 ? backend_->backends_.size()
                                              : extra_split_backends;
  // First store the work item count and reserve memory from them.
  dataready_.store(work_items, std::memory_order_relaxed);
  children_.reserve(work_items);
  size_t i = start_index;
  // First send work to backends which get extra work.
  int split_size = split_size_per_backend + 1;
  for (; i != end_index; i++) {
    assert(work_start != UsedBatchSize());
    size_t next_i = i + 1;
    size_t idx = backend_order[i].idx;
    size_t work_end =
        next_i == end_index && !are_all_splits_full
            ? work_start + last_split_size + (split_size - 1) * step
            : work_start + split_size * step;
    work_end = std::min(work_end, UsedBatchSize());
    children_.emplace_back(this, work_start, work_end);
    backend_->backends_[idx].Enqueue(&children_.back());
    work_start = work_end;
  }
  // Queue remaining work items which don't get extra work.
  split_size--;
  if (split_size > 0) {
    for (; i != backend_order.size(); i++) {
      assert(work_start != UsedBatchSize());
      size_t work_end = work_start + split_size * step;
      size_t idx = backend_order[i].idx;
      work_end = std::min(work_end, UsedBatchSize());
      children_.emplace_back(this, work_start, work_end);
      backend_->backends_[idx].Enqueue(&children_.back());
      work_start = work_end;
    }
  }
  assert(work_start == UsedBatchSize());
  assert(work_items == (int)children_.size());
  // Wait until all backends complete their work.
  if (!backend_->uses_cpu_backend_) {
    first_done_.wait(false, std::memory_order_acquire);
    // Use spinloop to reduce wake-up latency.
    while (dataready_.load(std::memory_order_acquire) != 0) {
      SpinloopPause();
    }
  } else {
    last_done_.wait(0, std::memory_order_acquire);
  }
}

class DemuxingBackendFactory : public BackendFactory {
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    const std::string backend_options_string =
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId);
    OptionsDict backend_options;
    backend_options.AddSubdictFromString(backend_options_string);

    std::string net_path =
        options.Get<std::string>(SharedBackendParams::kWeightsId);
    std::optional<WeightsFile> weights = LoadWeights(net_path);
    return std::make_unique<DemuxingBackend>(weights, options, backend_options);
  }

  std::string_view GetName() const override {
    using namespace std::string_view_literals;
    return "demux"sv;
  }

  int GetPriority() const override { return -1001; }
};

BackendManager::Register register_demux(
    std::make_unique<DemuxingBackendFactory>());

}  // namespace
}  // namespace lczero
