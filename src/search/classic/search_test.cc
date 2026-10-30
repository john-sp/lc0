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

#include "search/classic/search.h"

#include <cmath>
#include <cstdio>

#include "gtest/gtest.h"
#include "neural/encoder.h"
#include "neural/shared_params.h"
#include "trainingdata/trainingdata.h"

namespace lczero {
namespace classic {
namespace {

class UniformComputation : public BackendComputation {
 public:
  size_t UsedBatchSize() const override { return results_.size(); }
  AddInputResult AddInput(const EvalPosition&, EvalResultPtr result) override {
    results_.push_back(result);
    return ENQUEUED_FOR_EVAL;
  }
  void ComputeBlocking() override {
    for (auto result : results_) {
      if (result.q) *result.q = 0.0f;
      if (result.d) *result.d = 0.0f;
      if (result.m) *result.m = 0.0f;
      if (!result.p.empty()) {
        std::fill(result.p.begin(), result.p.end(), 1.0f / result.p.size());
      }
    }
  }

 private:
  std::vector<EvalResultPtr> results_;
};

class UniformBackend : public Backend {
 public:
  BackendAttributes GetAttributes() const override {
    return {false, true, true, 1, 32, 1024};
  }
  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<UniformComputation>();
  }
  std::optional<EvalResult> GetCachedEvaluation(const EvalPosition& pos) override {
    if (!cached) return std::nullopt;
    return EvalResult{0.0f, 0.0f, 0.0f,
                      std::vector<float>(pos.legal_moves.size(),
                                         1.0f / pos.legal_moves.size())};
  }
  bool cached = false;
};

class PlayoutLimit : public SearchStopper {
 public:
  explicit PlayoutLimit(int limit) : limit_(limit) {}
  bool ShouldStop(const IterationStats& stats, StoppersHints*) override {
    EXPECT_GE(stats.total_nodes, 0);
    EXPECT_LE(stats.total_nodes, initial_visits + stats.nodes_since_movestart);
    return stats.nodes_since_movestart >= limit_;
  }
  int initial_visits = 0;

 private:
  int limit_;
};

class ClassicSearchTest : public testing::Test {
 protected:
  ClassicSearchTest() {
    SharedBackendParams::Populate(&parser_);
    SearchParams::Populate(&parser_);
    options_ = parser_.GetMutableOptions();
    options_->Set<float>(SharedBackendParams::kPolicySoftmaxTemp, 1.0f);
    options_->Set<int>(SearchParams::kMiniBatchSizeId, 4);
    options_->Set<float>(SearchParams::kCpuctFactorId, 0.0f);
    options_->Set<float>(SearchParams::kSingleChildForcedBoostId, 0.0f);
    tree_.ResetToPosition(ChessBoard::kStartposFen, {});
    moves_ = tree_.GetPositionHistory().Last().GetBoard().GenerateLegalMoves();
  }

  static void AddVisits(Node* node, int n, float q = 0.0f) {
    node->IncrementNInFlight(n);
    node->FinalizeScoreUpdate(q, 0.0f, 0.0f, n);
  }

  void ExpandRoot() {
    auto* root = tree_.GetCurrentHead();
    root->CreateEdges(moves_);
    for (auto edge : root->Edges()) edge.edge()->SetP(1.0f / moves_.size());
    AddVisits(root, 1);
  }

  std::unique_ptr<Search> MakeSearch(const MoveList& filter = {}, int limit = 8) {
    auto stopper = std::make_unique<PlayoutLimit>(limit);
    stopper->initial_visits = tree_.GetCurrentHead()->GetN();
    return std::make_unique<Search>(
        tree_, &backend_, std::make_unique<UciResponderForwarder>(), filter,
        std::chrono::steady_clock::now(), std::move(stopper), false, false,
        *options_, nullptr);
  }

  OptionsParser parser_;
  OptionsDict* options_;
  NodeTree tree_;
  UniformBackend backend_;
  MoveList moves_;
};

TEST_F(ClassicSearchTest, TrainingChangesAreOptIn) {
  SearchParams params(*options_);
  EXPECT_FALSE(params.UsePolicyPostProcessing());
  EXPECT_EQ(params.GetForcedExplorationVisits(), 0);
}

TEST_F(ClassicSearchTest, DisabledProcessingKeepsRawVisitsWithCachedPolicy) {
  ExpandRoot();
  auto* root = tree_.GetCurrentHead();
  AddVisits(root->Edges().GetOrSpawnNode(root), 3);
  AddVisits(root, 3);
  backend_.cached = true;
  auto search = MakeSearch();
  auto distribution = search->GetVisitDistribution(moves_);
  EXPECT_FLOAT_EQ(std::get<0>(distribution[0]), 3.0f);
  for (size_t i = 1; i < distribution.size(); ++i) {
    EXPECT_FLOAT_EQ(std::get<0>(distribution[i]), 0.0f);
  }
}

TEST_F(ClassicSearchTest, LosingTargetsIgnoreUnvisitedUtility) {
  options_->Set<bool>(SearchParams::kUsePolicyPostProcessingId, true);
  ExpandRoot();
  auto* root = tree_.GetCurrentHead();
  auto edge = root->Edges();
  AddVisits(edge.GetOrSpawnNode(root), 10, -0.5f);
  ++edge;
  AddVisits(edge.GetOrSpawnNode(root), 10, -0.6f);
  AddVisits(root, 20);
  auto search = MakeSearch();
  auto distribution = search->GetVisitDistribution(moves_);
  double n0 = std::get<0>(distribution[0]);
  double n1 = std::get<0>(distribution[1]);
  EXPECT_NEAR(n0 / (n0 + n1), 0.8754548, 1e-5);
}

TEST_F(ClassicSearchTest, NoEvaluatedChildrenProduceFiniteTargets) {
  options_->Set<bool>(SearchParams::kUsePolicyPostProcessingId, true);
  ExpandRoot();
  backend_.cached = true;
  auto search = MakeSearch();
  auto distribution = search->GetVisitDistribution(moves_);
  float sum = 0.0f;
  for (auto [n, p] : distribution) {
    EXPECT_TRUE(std::isfinite(n));
    sum += n;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-6);
}

TEST_F(ClassicSearchTest, FilterMaySkipTheFirstAndLastPolicyEdges) {
  options_->Set<int>(SearchParams::kForcedExplorationVisitsId, 220);
  // Use a fresh root so the first root evaluation also exercises initialization.
  auto search = MakeSearch({moves_[moves_.size() / 2]}, 12);
  search->RunBlocking(1);
  EXPECT_EQ(search->GetBestMove().first, moves_[moves_.size() / 2]);
  EXPECT_EQ(tree_.GetCurrentHead()->GetNInFlight(), 0);
}

TEST_F(ClassicSearchTest, ReusedRootReceivesForcedExploration) {
  ExpandRoot();
  auto* root = tree_.GetCurrentHead();
  auto edge = root->Edges();
  edge.edge()->SetP(0.9f);
  auto* first = edge.GetOrSpawnNode(root);
  AddVisits(first, 1);
  auto history = tree_.GetPositionHistory();
  history.Append(edge.GetMove());
  auto child_moves = history.Last().GetBoard().GenerateLegalMoves();
  first->CreateEdges(child_moves);
  for (auto child : first->Edges()) {
    child.edge()->SetP(1.0f / child_moves.size());
  }
  for (++edge; edge; ++edge) edge.edge()->SetP(0.1f / (moves_.size() - 1));
  AddVisits(root, 1);
  options_->Set<int>(SearchParams::kForcedExplorationVisitsId, 220);
  options_->Set<float>(SearchParams::kCpuctId, 0.0f);
  auto search = MakeSearch({}, 4);
  search->RunBlocking(1);
  int visited = 0;
  for (auto child : root->Edges()) visited += child.GetN() > 0;
  EXPECT_GT(visited, 1);
  EXPECT_EQ(root->GetNInFlight(), 0);
}

TEST_F(ClassicSearchTest, ForcedVisitEstimateDoesNotInflateCompletedNodes) {
  ExpandRoot();
  auto* root = tree_.GetCurrentHead();
  auto edge = root->Edges();
  for (int i = 0; i < 2; ++i, ++edge) {
    auto* child = edge.GetOrSpawnNode(root);
    auto history = tree_.GetPositionHistory();
    history.Append(edge.GetMove());
    auto child_moves = history.Last().GetBoard().GenerateLegalMoves();
    child->CreateEdges(child_moves);
    for (auto child_edge : child->Edges()) {
      child_edge.edge()->SetP(1.0f / child_moves.size());
    }
    AddVisits(child, i == 0 ? 1000 : 1, i == 0 ? 0.0f : -0.0001f);
  }
  AddVisits(root, 1001);
  options_->Set<int>(SearchParams::kForcedExplorationVisitsId, 1);
  // PlayoutLimit also checks that adjusted total_nodes never exceeds the
  // number of completed visits, even when reconstructed visits exceed N.
  auto search = MakeSearch({}, 1);
  search->RunBlocking(1);
}

TEST_F(ClassicSearchTest, TemperatureHandlesUnbalanceableScores) {
  ExpandRoot();
  auto* root = tree_.GetCurrentHead();
  auto edge = root->Edges();
  edge.edge()->SetP(0.01f);
  AddVisits(edge.GetOrSpawnNode(root), 20, -0.5f);
  ++edge;
  edge.edge()->SetP(0.001f);
  AddVisits(edge.GetOrSpawnNode(root), 1, 0.5f);
  AddVisits(root, 21);
  options_->Set<int>(SearchParams::kForcedExplorationVisitsId, 220);
  options_->Set<float>(SearchParams::kTemperatureId, 1.0f);
  auto search = MakeSearch();
  auto move = search->GetBestMove().first;
  EXPECT_TRUE(move == moves_[0] || move == moves_[1]);
}

TEST_F(ClassicSearchTest, TemperatureHandlesNoEligibleVisitAboveOffset) {
  ExpandRoot();
  options_->Set<int>(SearchParams::kForcedExplorationVisitsId, 220);
  options_->Set<float>(SearchParams::kTemperatureId, 1.0f);
  options_->Set<float>(SearchParams::kTemperatureVisitOffsetId, -10.0f);
  auto search = MakeSearch();
  auto move = search->GetBestMove().first;
  EXPECT_NE(std::find(moves_.begin(), moves_.end(), move), moves_.end());
}

TEST_F(ClassicSearchTest, SingleUnvisitedMoveWritesProbabilityOneWithoutCache) {
  tree_.ResetToPosition("8/8/8/8/8/1rk5/8/K7 w - - 0 1", {});
  moves_ = tree_.GetPositionHistory().Last().GetBoard().GenerateLegalMoves();
  ASSERT_EQ(moves_.size(), 1);
  ExpandRoot();
  options_->Set<bool>(SearchParams::kUsePolicyPostProcessingId, true);
  auto search = MakeSearch();
  V6TrainingDataArray data(FillEmptyHistory::NO, FillEmptyHistory::NO,
      pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE);
  data.Add(tree_.GetCurrentHead(), tree_.GetPositionHistory(), {}, {}, false,
           moves_[0], moves_[0], search->GetVisitDistribution(moves_), moves_,
           std::nullopt);
  const auto filename = testing::TempDir() + "lc0-single-unvisited-move.gz";
  {
    TrainingDataWriter writer(filename);
    data.Write(&writer, GameResult::DRAW, false);
  }
  V6TrainingData chunk;
  auto file = gzopen(filename.c_str(), "rb");
  ASSERT_NE(file, nullptr);
  EXPECT_EQ(gzread(file, &chunk, sizeof(chunk)), sizeof(chunk));
  gzclose(file);
  std::remove(filename.c_str());
  EXPECT_FLOAT_EQ(chunk.probabilities[MoveToNNIndex(moves_[0], 0)], 1.0f);
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
