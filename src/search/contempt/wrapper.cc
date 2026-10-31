/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include <iomanip>
#include <limits>
#include <sstream>

#include "chess/gamestate.h"
#include "neural/shared_params.h"
#include "search/classic/stoppers/factory.h"
#include "search/contempt/search.h"
#include "search/register.h"
#include "search/search.h"
#include "utils/trace.h"

namespace lczero {
namespace contempt {
namespace {

const OptionId kThreadsOptionId{
    {.long_flag = "threads",
     .uci_option = "Threads",
     .help_text =
         "Number of (CPU) worker threads to use, 0 for the backend default.",
     .short_flag = 't',
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kClearTree{
    {.long_flag = "",
     .uci_option = "ClearTree",
     .help_text = "Clear the tree before the next search.",
     .visibility = OptionId::kProOnly}};

class ContemptSearch : public SearchBase {
 public:
  ContemptSearch(UciResponder* responder, const OptionsDict* options)
      : SearchBase(responder), options_(options) {}
  ~ContemptSearch() { search_.reset(); }

  void SetBackend(Backend* backend) override {
    if (backend_ != backend) {
      search_.reset();
      tt_.clear();
      tree_.reset();
      reuse_key_.clear();
    }
    SearchBase::SetBackend(backend);
  }

 private:
  void NewGame() override;
  void SetPosition(const GameState& pos) override;
  void StartSearch(const GoParams&) override;
  void StartClock() override {
    move_start_time_ = std::chrono::steady_clock::now();
  }
  void WaitSearch() override {
    LOGFILE << "Waiting for the search.";
    if (search_) search_->Wait();
  }
  void StopSearch() override {
    if (search_) search_->Stop();
  }
  void AbortSearch() override {
    if (search_) search_->Abort();
  }

  const OptionsDict* options_;
  std::unique_ptr<classic::TimeManager> time_manager_;
  std::unique_ptr<Search> search_;
  std::unique_ptr<NodeTree> tree_;
  TranspositionTable tt_;
  std::string reuse_key_;
  std::optional<std::chrono::steady_clock::time_point> move_start_time_;
};

MoveList StringsToMovelist(const std::vector<std::string>& moves,
                           const ChessBoard& board) {
  MoveList result;
  if (moves.size()) {
    result.reserve(moves.size());
    const auto legal_moves = board.GenerateLegalMoves();
    for (const auto& move : moves) {
      const Move m = board.ParseMove(move);
      if (std::find(legal_moves.begin(), legal_moves.end(), m) !=
          legal_moves.end()) {
        result.emplace_back(m);
      }
    }
    if (result.empty()) throw Exception("No legal searchmoves.");
  }
  return result;
}

void ContemptSearch::NewGame() {
  LCTRACE_FUNCTION_SCOPE;
  LOGFILE << "New game.";
  search_.reset();
  tt_.clear();
  tree_.reset();
  reuse_key_.clear();
  time_manager_ = classic::MakeTimeManager(*options_);
}

void ContemptSearch::SetPosition(const GameState& pos) {
  LCTRACE_FUNCTION_SCOPE;
  search_.reset();
  if (!tree_) tree_ = std::make_unique<NodeTree>();
  const bool is_same_game = tree_->ResetToPosition(pos);
  LOGFILE << "Tree reset to a new position.";
  if (!is_same_game) {
    tt_.clear();
    time_manager_ = classic::MakeTimeManager(*options_);
  }
}

void ContemptSearch::StartSearch(const GoParams& params) {
  LCTRACE_FUNCTION_SCOPE;
  search_.reset();
  // Stored values and frozen sampling policies must belong to one evaluation
  // configuration. Clearing the TT alone would leave the old graph reachable.
  const SearchParams config(*options_);
  const auto mode = config.ResolveContemptMode(
      tree_->HeadPosition().IsBlackToMove(), params.infinite, params.ponder);
  std::ostringstream key;
  key << std::setprecision(std::numeric_limits<float>::max_digits10)
      << backend_ << ' ' << syzygy_tb_ << ' ' << static_cast<int>(mode) << ' '
      << config.GetWDLRescaleRatio() << ' ' << config.GetWDLRescaleDiff() << ' '
      << config.GetWDLMaxS() << ' ' << config.GetContemptModeTB() << ' '
      << config.GetCacheHistoryLength() << ' '
      << static_cast<int>(config.GetHistoryFill()) << ' '
      << config.GetSyzygyFastPlay() << ' ' << config.GetTwoFoldDraws() << ' '
      << config.GetStickyEndgames() << ' ' << config.GetDrawScore() << ' '
      << config.GetPolicySoftmaxTemp() << ' ' << config.GetNoiseEpsilon() << ' '
      << config.GetNoiseAlpha() << ' ' << config.GetUseUncertaintyWeighting() << ' '
      << config.GetUncertaintyWeightingMidPoint() << ' '
      << config.GetUncertaintyWeightingMinusExponent() << ' '
      << config.GetUncertaintyWeightingScale() << ' '
      << config.GetUncertaintyWeightingBase() << ' '
      << config.GetUncertaintyWeightingCap() << ' ' << config.GetScLimit() << ' '
      << static_cast<int>(config.GetHybridRatioMode()) << ' '
      << config.GetHybridSamplingRatio() << ' ' << config.GetHybridMinRatio() << ' '
      << config.GetHybridMaxRatio() << ' ' << config.GetHybridScalingFactor() << ' '
      << config.GetHybridShapeParam1() << ' ' << config.GetHybridShapeParam2();
  for (const auto& [visits, ratio] : config.GetHybridRatioSchedule()) {
    key << ' ' << visits << ':' << ratio;
  }
  const std::string reuse_key = key.str();
  if (!reuse_key_.empty() && reuse_key_ != reuse_key) {
    tt_.clear();
    tree_->TrimTreeAtHead();
  }
  reuse_key_ = reuse_key;

  auto forwarder =
      std::make_unique<NonOwningUciRespondForwarder>(uci_responder_);
  if (options_->Get<Button>(kClearTree).TestAndReset()) {
    tt_.clear();
    tree_->TrimTreeAtHead();
    LOGFILE << "Tree cleared.";
  }

  const auto cache_size =
      options_->Get<int>(SharedBackendParams::kNNCacheSizeId);
  // Conservatively include a shared position and frozen CDF per visited edge.
  const size_t kAvgNodeSize =
      sizeof(Node) + sizeof(LowNode) +
      classic::MemoryWatchingStopper::kAvgMovesPerPosition *
          (sizeof(Edge) + sizeof(double));
  const size_t kAvgCacheItemSize =
      4 * sizeof(float) + sizeof(std::unique_ptr<float[]>) +
      sizeof(float[classic::MemoryWatchingStopper::kAvgMovesPerPosition]);
  size_t total_memory =
      tree_.get()->GetCurrentHead()->GetCompletedVisits() * kAvgNodeSize +
      (sizeof(TranspositionTable::value_type) + 1) * tt_.bucket_count() +
      cache_size * kAvgCacheItemSize;
  auto stopper = time_manager_->GetStopper(
      params, tree_.get()->HeadPosition(), total_memory, kAvgNodeSize,
      tree_.get()->GetCurrentHead()->GetCompletedVisits());
  search_ = std::make_unique<Search>(
      *tree_, backend_, std::move(forwarder),
      StringsToMovelist(params.searchmoves, tree_->HeadPosition().GetBoard()),
      *move_start_time_, std::move(stopper), params.infinite, params.ponder,
      *options_, &tt_, syzygy_tb_);

  LOGFILE << "Timer started at "
          << FormatTime(SteadyClockToSystemClock(*move_start_time_));
  search_->StartThreads(options_->Get<int>(kThreadsOptionId));
}

class ContemptSearchFactory : public SearchFactory {
  std::string_view GetName() const override { return "sc"; }
  std::unique_ptr<SearchBase> CreateSearch(
      UciResponder* responder, const OptionsDict* options) const override {
    LCTRACE_FUNCTION_SCOPE;
    return std::make_unique<ContemptSearch>(responder, options);
  }

  void PopulateParams(OptionsParser* parser) const override {
    parser->Add<IntOption>(kThreadsOptionId, 0, 128) = 0;
    SearchParams::Populate(parser);
    classic::PopulateTimeManagementOptions(classic::RunType::kUci, parser);

    parser->Add<ButtonOption>(kClearTree);
  }
};

REGISTER_SEARCH(ContemptSearchFactory);

}  // namespace
}  // namespace contempt
}  // namespace lczero
