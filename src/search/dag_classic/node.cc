/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include "search/dag_classic/node.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <sstream>
#include <thread>

#include "search/dag_classic/search.h"
#include "utils/trace.h"

namespace lczero {
namespace dag_classic {

/////////////////////////////////////////////////////////////////////////
// Edge
/////////////////////////////////////////////////////////////////////////

Move Edge::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Flip();
  return m;
}

// Policy priors (P) are stored in a compressed 16-bit format.
//
// Source values are 32-bit floats:
// * bit 31 is sign (zero means positive)
// * bit 30 is sign of exponent (zero means nonpositive)
// * bits 29..23 are value bits of exponent
// * bits 22..0 are significand bits (plus a "virtual" always-on bit: s ∈ [1,2))
// The number is then sign * 2^exponent * significand, usually.
// See https://www.h-schmidt.net/FloatConverter/IEEE754.html for details.
//
// In compressed 16-bit value we store bits 27..12:
// * bit 31 is always off as values are always >= 0
// * bit 30 is always off as values are always < 2
// * bits 29..28 are only off for values < 4.6566e-10, assume they are always on
// * bits 11..0 are for higher precision, they are dropped leaving only 11 bits
//     of precision
//
// When converting to compressed format, bit 11 is added to in order to make it
// a rounding rather than truncation.
//
// Out of 65556 possible values, 2047 are outside of [0,1] interval (they are in
// interval (1,2)). This is fine because the values in [0,1] are skewed towards
// 0, which is also exactly how the components of policy tend to behave (since
// they add up to 1).

// If the two assumed-on exponent bits (3<<28) are in fact off, the input is
// rounded up to the smallest value with them on. We accomplish this by
// subtracting the two bits from the input and checking for a negative result
// (the subtraction works despite crossing from exponent to significand). This
// is combined with the round-to-nearest addition (1<<11) into one op.
void Edge::SetP(float p) {
  assert(0.0f <= p && p <= 1.0f);
  constexpr int32_t roundings = (1 << 11) - (3 << 28);
  int32_t tmp;
  std::memcpy(&tmp, &p, sizeof(float));
  tmp += roundings;
  p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
}

float Edge::GetP() const {
  // Reshift into place and set the assumed-set exponent bits.
  uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
  float ret;
  std::memcpy(&ret, &tmp, sizeof(uint32_t));
  return ret;
}

std::string Edge::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.ToString(true) << " p_: " << p_
      << " GetP: " << GetP();
  return oss.str();
}

void Edge::FromMovelist(Edge* dst, const MoveList& moves) {
  std::transform(moves.begin(), moves.end(), dst, [](Move move) {
    Edge edge;
    edge.move_ = move;
    return edge;
  });
}

/////////////////////////////////////////////////////////////////////////
// LowNode + Node
/////////////////////////////////////////////////////////////////////////

void Node::Trim() {
  wl_ = 0.0f;

  UnsetLowNode();
  // sibling_

  d_ = 0.0f;
  m_ = 0.0f;
  weight_ = 0.0;
  n_in_flight_.store(0, std::memory_order_release);

  // edge_

  // index_

  terminal_type_ = Terminal::NonTerminal;
  lower_bound_ = GameResult::BLACK_WON;
  upper_bound_ = GameResult::WHITE_WON;
  repetition_ = false;
}

LowNode::~LowNode() {
  if (solid_edges_) {
    NGC::Instance().AddToGcQueue(child_.solid_);
  } else {
    NGC::Instance().AddToGcQueue(child_.first_);
  }
}

Node::~Node() {
  NGC::Instance().AddToGcQueue(sibling_);
  UnsetLowNode();
}

Node* Node::GetChild() const {
  if (!low_node_) return nullptr;
  return low_node_->GetChild();
}

bool Node::HasChildren() const { return low_node_ && low_node_->HasChildren(); }

float Node::GetVisitedPolicy() const {
  float sum = 0.0f;
  for (auto* node : VisitedNodes()) sum += node->GetP();
  return sum;
}

uint32_t Node::GetNInFlight() const {
  return n_in_flight_.load(std::memory_order_acquire);
}

float Node::GetChildrenVisits() const {
  return low_node_ ? low_node_->GetChildrenVisits() : 0.0f;
}

float Node::GetTotalVisits() const {
  return low_node_ ? low_node_->GetN() : 0.0f;
}

const Edge& LowNode::GetEdgeAt(uint16_t index) const {
  if (solid_edges_) {
    return child_.solid_->GetEdges(num_edges_)[index];
  } else {
    return child_.first_->GetEdges()[index];
  }
}

Node* LowNode::GetChildAt(uint16_t index) const {
  if (solid_edges_) {
    return &child_.solid_->GetChild()[index];
  } else {
    Node* node = child_.first_->GetChild();
    for (size_t i = 0; i < index && node; ++i) {
      node = node->GetSibling()->get();
    }
    return node;
  }
}

std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " <Node> This:" << this << " LowNode:" << low_node_.get()
      << " Index:" << index_ << " Move:" << GetMove().ToString(true)
      << " Sibling:" << sibling_.get() << " P:" << GetP() << " WL:" << wl_
      << " D:" << d_ << " M:" << m_ << " N:" << weight_
      << " N_:" << GetNInFlight()
      << " Term:" << static_cast<int>(terminal_type_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2;
  return oss.str();
}

std::string LowNode::DebugString() const {
  std::ostringstream oss;
  oss << " <LowNode> This:" << this << " Edges:" << GetEdges()
      << " NumEdges:" << static_cast<int>(num_edges_) << " Child:" << GetChild()
      << " WL:" << wl_ << " D:" << d_ << " M:" << m_ << " N:" << weight_
      << " NP:" << num_parents_ << " Term:" << static_cast<int>(terminal_type_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2;
  return oss.str();
}

void Edge::SortEdges(Edge* edges, int num_edges) {
  // Sorting on raw p_ is the same as sorting on GetP() as a side effect of
  // the encoding, and its noticeably faster.
  std::sort(edges, (edges + num_edges),
            [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
}

void LowNode::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
  }

  assert(WLDMInvariantsHold());
}

void LowNode::MakeNotTerminal(const Node* node) {
  assert(child_.first_);
  if (!IsTerminal()) return;

  terminal_type_ = Terminal::NonTerminal;
  lower_bound_ = GameResult::BLACK_WON;
  upper_bound_ = GameResult::WHITE_WON;
  weight_ = 0.0;
  wl_ = 0.0;
  d_ = 0.0;
  m_ = 0.0;

  // Include children too.
  if (node->GetNumEdges() > 0) {
    for (const auto& child : node->Edges()) {
      auto weight = child.GetWeight();
      if (weight > 0) {
        weight_ += weight;
        // Flip Q for opponent.
        // Default values don't matter as n is > 0.
        wl_ += child.GetWL(0.0f) * weight;
        d_ += child.GetD(0.0f) * weight;
        m_ += child.GetM(0.0f) * weight;
      }
    }

    // Recompute with current eval (instead of network's) and children's eval.
    wl_ /= weight_;
    d_ /= weight_;
    m_ /= weight_;
  }

  assert(WLDMInvariantsHold());
}

void LowNode::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

uint8_t Node::GetNumEdges() const {
  return low_node_ ? low_node_->GetNumEdges() : 0;
}

void Node::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
    // Terminal losses have no uncertainty and no reason for their U value to
    // be comparable to another non-loss choice. Force this by clearing the
    // policy.
    SetP(0.0f);
  }

  assert(WLDMInvariantsHold());
}

void Node::MakeNotTerminal(bool also_low_node) {
  // At least one of node and low node pair needs to be a terminal.
  if (!IsTerminal() &&
      (!also_low_node || !low_node_ || !low_node_->IsTerminal()))
    return;

  terminal_type_ = Terminal::NonTerminal;
  repetition_ = false;
  if (low_node_) {  // Two-fold or derived terminal.
    // Revert low node first.
    if (also_low_node && low_node_) low_node_->MakeNotTerminal(this);

    auto [lower_bound, upper_bound] = low_node_->GetBounds();
    lower_bound_ = -upper_bound;
    upper_bound_ = -lower_bound;
    weight_ = low_node_->GetWeight();
    wl_ = -low_node_->GetWL();
    d_ = low_node_->GetD();
    m_ = low_node_->GetM() + 1;
  } else {  // Real terminal.
    lower_bound_ = GameResult::BLACK_WON;
    upper_bound_ = GameResult::WHITE_WON;
    weight_ = 0.0f;
    wl_ = 0.0f;
    d_ = 0.0f;
    m_ = 0.0f;
  }

  assert(WLDMInvariantsHold());
}

void Node::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

bool Node::TryStartScoreUpdate() {
  if (weight_ > 0) {
    n_in_flight_.fetch_add(1, std::memory_order_acq_rel);
    return true;
  } else {
    uint32_t expected_n_if_flight_ = 0;
    return n_in_flight_.compare_exchange_strong(expected_n_if_flight_, 1,
                                                std::memory_order_acq_rel);
  }
}

void Node::CancelScoreUpdate(uint32_t multivisit) {
  assert(GetNInFlight() >= (uint32_t)multivisit);
  n_in_flight_.fetch_sub(multivisit, std::memory_order_acq_rel);
}

double LowNode::FinalizeScoreUpdate(double v, double d, float m,
                                    double multiweight) {
  assert(child_.first_);
  // Increment N.
  weight_ += multiweight;

  // Recompute Q.
  double divisor = 1.0 / GetWeight();
  wl_ += multiweight * (v - wl_) * divisor;
  d_ += multiweight * (d - d_) * divisor;
  m_ += multiweight * (m - m_) * static_cast<float>(divisor);

  assert(WLDMInvariantsHold());
  return divisor;
}

void LowNode::AdjustForTerminal(double v, double d, float m, double divisor,
                                double multiweight) {
  assert(multiweight <= weight_);

  // Recompute Q.
  wl_ += multiweight * v * divisor;
  d_ += multiweight * d * divisor;
  m_ += multiweight * m * static_cast<float>(divisor);

  assert(WLDMInvariantsHold());
}

double Node::FinalizeScoreUpdate(double v, double d, float m,
                                 double multiweight) {
  // Increment N.
  weight_ += multiweight;

  // Recompute Q.
  double divisor = 1.0 / weight_;
  wl_ += multiweight * (v - wl_) * divisor;
  d_ += multiweight * (d - d_) * divisor;
  m_ += multiweight * (m - m_) * static_cast<float>(divisor);

  assert(WLDMInvariantsHold());
  // Decrement virtual loss.
  auto old = n_in_flight_.fetch_sub(1, std::memory_order_acq_rel);
  assert(old > 0);
  return divisor;
}

void Node::AdjustForTerminal(double v, double d, float m, double divisor,
                             double multiweight) {
  assert(multiweight <= weight_);

  // Recompute Q.
  wl_ += multiweight * v * divisor;
  d_ += multiweight * d * divisor;
  m_ += multiweight * m * static_cast<float>(divisor);

  assert(WLDMInvariantsHold());
}

void Node::IncrementNInFlight(uint32_t multivisit) {
  n_in_flight_.fetch_add(multivisit, std::memory_order_acq_rel);
}

Node::Node(Node&& node)
    : low_node_(std::move(node.low_node_)),
      wl_(node.wl_),
      d_(node.d_),
      weight_(node.weight_),
      sibling_(nullptr),
      m_(node.m_),
      n_in_flight_(node.n_in_flight_.load(std::memory_order_relaxed)),
      edge_(node.edge_),
      index_(node.index_),
      terminal_type_(node.terminal_type_),
      lower_bound_(node.lower_bound_),
      upper_bound_(node.upper_bound_),
      repetition_(node.repetition_) {
  assert(GetN() == 0 || IsTerminal() || GetLowNode());
}

Node& Node::operator=(Node&& node) {
  low_node_ = std::move(node.low_node_);
  wl_ = node.wl_;
  d_ = node.d_;
  weight_ = node.weight_;
  sibling_ = nullptr;
  m_ = node.m_;
  n_in_flight_.store(node.n_in_flight_.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
  edge_ = node.edge_;
  index_ = node.index_;
  terminal_type_ = node.terminal_type_;
  lower_bound_ = node.lower_bound_;
  upper_bound_ = node.upper_bound_;
  repetition_ = node.repetition_;
  assert(GetN() == 0 || IsTerminal() || GetLowNode());
  return *this;
}

std::unique_ptr<LowNode::ChildAndEdges> LowNode::ChildAndEdges::FromMovelist(
    const MoveList& moves, uint16_t index) {
  std::unique_ptr<ChildAndEdges> child(ChildAndEdges::Allocate(moves.size()));

  Edge::FromMovelist(child->GetEdges(), moves);

  std::construct_at(child->GetChild(), child->GetEdges()[index], index);

  return child;
}

std::unique_ptr<LowNode::ChildAndEdges> LowNode::ChildAndEdges::FromNode(
    Node&& node, uint8_t num_edges) {
  std::unique_ptr<ChildAndEdges> child(ChildAndEdges::Allocate(num_edges));

  std::construct_at(child->GetChild(), std::move(node));

  return child;
}

LowNode::LowNode(const LowNode& node, Move saved_child)
    : wl_(node.wl_),
      d_(node.d_),
      weight_(node.weight_),
      child_(ChildAndEdges::Allocate(node.num_edges_)),
      m_(node.m_),
      num_edges_(node.num_edges_),
      terminal_type_(node.terminal_type_),
      lower_bound_(node.lower_bound_),
      upper_bound_(node.upper_bound_),
      solid_edges_(false) {
  std::copy(node.GetEdges(), node.GetEdges() + node.num_edges_,
            child_.first_->GetEdges());
  Node* src = node.child_.first_->GetChild();
  for (; src; src = src->GetSibling()->get()) {
    if (src->GetMove() == saved_child) {
      break;
    }
  }
  if (src) {
    std::construct_at(child_.first_->GetChild(), std::move(*src));
  } else {
    auto edge = std::find_if(node.child_.first_->GetEdges(),
                             node.child_.first_->GetEdges() + node.num_edges_,
                             [saved_child](const Edge& edge) {
                               return edge.GetMove() == saved_child;
                             });
    assert(edge != node.child_.first_->GetEdges() + node.num_edges_);
    // If the saved child is not found, construct a default Node.
    std::construct_at(child_.first_->GetChild(), *edge,
                      edge - node.child_.first_->GetEdges());
  }
}

LowNode::SolidChildren::~SolidChildren() {
  Node* node = GetChild();
  Node* next = node->GetSibling()->get();
  for (; node; node = next, next = next ? next->GetSibling()->get() : nullptr) {
    // Disconnect the sibling pointer which doesn't own the memory.
    node->GetSibling()->release();
    // Call destructor to clean nodes.
    std::destroy_at(node);
  }
}

LowNode::ChildAndEdges* LowNode::ChildAndEdges::Allocate(size_t count) {
  // Allocates aligned memory for variable size class.
  size_t size = sizeof(ChildAndEdges) + count * sizeof(Edge);
  size = (size + Node::kAlignment - 1) & ~(Node::kAlignment - 1);
#if defined(_MSC_VER)
  return reinterpret_cast<ChildAndEdges*>(
      _aligned_malloc(size, Node::kAlignment));
#else
  return reinterpret_cast<ChildAndEdges*>(
      std::aligned_alloc(Node::kAlignment, size));
#endif
}

void LowNode::ChildAndEdges::operator delete(ChildAndEdges* ptr,
                                             std::destroying_delete_t) {
  // Custom delete operator to free aligned memory allocations.
  std::destroy_at(ptr);
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

LowNode::SolidChildren* LowNode::SolidChildren::Allocate(size_t count) {
  // Allocates aligned memory for variable size class.
  size_t size =
      sizeof(SolidChildren) + count * sizeof(Node) + count * sizeof(Edge);
  size = (size + Node::kAlignment - 1) & ~(Node::kAlignment - 1);
#if defined(_MSC_VER)
  return reinterpret_cast<SolidChildren*>(
      _aligned_malloc(size, Node::kAlignment));
#else
  return reinterpret_cast<SolidChildren*>(
      std::aligned_alloc(Node::kAlignment, size));
#endif
}

void LowNode::SolidChildren::operator delete(SolidChildren* ptr,
                                             std::destroying_delete_t) {
  // Custom delete operator to free aligned memory allocations.
  std::destroy_at(ptr);
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

std::unique_ptr<LowNode::SolidChildren> LowNode::SolidChildren::Make(
    uint8_t num_edges, Node* child, Edge* edges) {
  // Allocate and construct a SoldChildren from ChildAndEdges
  std::unique_ptr<SolidChildren> solid(SolidChildren::Allocate(num_edges));
  Node* dst = solid->GetChild();
  // Copy existing children and
  for (size_t i = 0; i < num_edges;
       ++i, child = child ? child->GetSibling()->get() : nullptr) {
    if (child) {
      assert(child->GetN() == 0 || child->IsTerminal() || child->GetLowNode());
      std::construct_at(&dst[i], std::move(*child));
    } else {
      std::construct_at(&dst[i], edges[i], i);
    }
    // Connect linked list for iteration.
    if (i + 1 < num_edges) {
      dst[i].GetSibling()->reset(&dst[i + 1]);
    }
  }
  // Copy edges.
  std::copy(edges, edges + num_edges, solid->GetEdges(num_edges));
  return solid;
}

LowNode::PointerChanges* LowNode::PointerChanges::Allocate(size_t count) {
  // Allocate variable size class to temporary iteration memory. There won't be
  // an destructor calls. All pointers must be non-owning because all memory
  // will be leaked.
  using Allocator = IterationMemoryAllocator<char>;
  size_t size = sizeof(PointerChanges) + count * sizeof(Node*) * 2;
  auto* rv = reinterpret_cast<PointerChanges*>(Allocator().allocate(size));
  rv->num_edges_ = 0;
  rv->capacity_ = count;
  return rv;
}

LowNode::PointerChanges* LowNode::MakeSolid() {
  if (solid_edges_) {
    return nullptr;
  }
  // Check that none of children are pending a NN evaluation.
  for (Node* node = child_.first_->GetChild(); node;
       node = node->GetSibling()->get()) {
    if (node->GetN() == 0 && node->GetNInFlight() > 0) {
      return nullptr;
    }
  }
  PointerChanges* result = PointerChanges::Allocate(num_edges_);
  // Collect old Node pointers to convert BackupPaths.
  for (Node* node = child_.first_->GetChild(); node;
       node = node->GetSibling()->get()) {
    assert(node->GetN() == 0 || node->IsTerminal() || node->GetLowNode());
    result->changes_[result->num_edges_++] = node;
  }
  auto solid = SolidChildren::Make(num_edges_, child_.first_->GetChild(),
                                   child_.first_->GetEdges());
  NGC::Instance().AddToGcQueue(child_.first_);
  child_.solid_ = solid.release();
  solid_edges_ = true;
  // Collect new pointers which are used to update BackupPaths.
  for (size_t i = 0; i < result->size(); ++i) {
    Node* node = &child_.solid_->GetChild()[i];
    assert(node->GetN() == 0 || node->IsTerminal() || node->GetLowNode());
    (*result)[i].new_ = node;
  }
  return result;
}

Node* Node::MakeSingleChild(Move move) {
  if (low_node_) {
    ReleaseChildrenExceptOne(move);
    return low_node_->GetChild();
  }
  return CreateSingleChildNode(move);
}

void LowNode::ReleaseChildrenExceptOne(Move move) {
  auto& ngc = NGC::Instance();
  // Stores node which will have to survive (or nullptr if it's not found).
  std::unique_ptr<ChildAndEdges> saved_node;
  // Handle first node as a special case.
  auto* first_node = GetChild();
  if (first_node->GetMove() == move) {
    if (solid_edges_) {
      saved_node = ChildAndEdges::FromNode(std::move(*first_node), num_edges_);
      ngc.AddToGcQueue(child_.solid_);
      child_.first_ = saved_node.release();
      solid_edges_ = false;
      return;
    } else {
      ngc.AddToGcQueue(*first_node->GetSibling());
      return;
    }
  }
  // Pointer to atomic_unique_ptr, so that we could move from it.
  for (auto node = first_node->GetSibling(); *node;
       node = (*node)->GetSibling()) {
    // If current node is the one that we have to save.
    if ((*node)->GetMove() == move) {
      // Save the node, and take the ownership from the unique_ptr.
      if (solid_edges_) {
        (*node)->GetSibling()->release();
        saved_node = ChildAndEdges::FromNode(std::move(**node), num_edges_);
        ngc.AddToGcQueue(child_.solid_);
      } else {
        *GetChild() = std::move(**node);
        saved_node.reset(child_.first_);
      }
      break;
    }
  }
  child_.first_ = saved_node.release();
  solid_edges_ = false;
  return;
}

void Node::ReleaseChildrenExceptOne(Move move) {
  // Sometime we have no graph yet or a reverted terminal without low node.
  if (low_node_) {
    if (low_node_->GetChild()->GetMove() != move ||
        low_node_->GetChild()->GetSibling()->get()) {
      // If low node isn't already a single child, we need to copy low_node to
      // avoid potential tt low node losing children.
      auto low = std::make_shared<LowNode>(*low_node_, move);
      UnsetLowNode();
      SetLowNode(low);
      return;
    }
  }
}

void Node::SetLowNode(std::shared_ptr<LowNode> low_node) {
  assert(!low_node_);
  low_node->AddParent();
  low_node_ = low_node;
}
void Node::UnsetLowNode() {
  if (low_node_) low_node_->RemoveParent();
  low_node_.reset();
}

#ifndef NDEBUG
namespace {
static Node::VisitorId::storage current_visitor_id = 0;
}

Node::VisitorId::VisitorId() {
  id_ = ++current_visitor_id;
  if (id_ == 0) id_ = ++current_visitor_id;
}

Node::VisitorId::~VisitorId() { assert(current_visitor_id == id_); }

bool LowNode::Visit(Node::VisitorId::type id) {
  if (visitor_id_ == id) return false;
  visitor_id_ = id;
  return true;
}

template <typename VisitorType, typename EdgeVisitorType>
static void TreeWalk(const Node* node, bool as_opponent,
                     Node::VisitorId::type id, VisitorType visitor,
                     EdgeVisitorType edge) {
  const std::shared_ptr<LowNode>& low_node = node->GetLowNode();
  if (!low_node || !low_node->Visit(id)) {
    return;
  }

  visitor(low_node.get(), as_opponent);

  for (auto& child_edge : node->Edges()) {
    auto child = child_edge.node();
    if (child == nullptr) {
      break;
    }
    edge(child, as_opponent, low_node.get());
  }

  for (auto& child_edge : node->Edges()) {
    auto child = child_edge.node();
    if (child == nullptr) {
      return;
    }
    TreeWalk(child, !as_opponent, id, visitor, edge);
  }
}

static std::string PtrToNodeName(const void* ptr) {
  std::ostringstream oss;
  oss << "n_" << ptr;
  return oss.str();
}

template <typename VisitorType, typename EdgeVisitorType>
static void TreeWalk(const Node* node, bool as_opponent, VisitorType visitor,
                     EdgeVisitorType edge) {
  Node::VisitorId id{};
  edge(node, as_opponent, nullptr);
  TreeWalk(node, !as_opponent, id, visitor, edge);
}

void LowNode::DotNodeString(std::ofstream& oss) const {
  oss << PtrToNodeName(this) << " ["
      << "shape=box";
  // Adjust formatting to limit node size.
  oss << std::fixed << std::setprecision(3);
  oss << ",label=\""     //
      << std::showpos    //
      << "WL=" << wl_    //
      << std::noshowpos  //
      << "\\lD=" << d_ << "\\lM=" << m_ << "\\lN=" << weight_ << "\\l\"";
  // Set precision for tooltip.
  oss << std::fixed << std::showpos << std::setprecision(5);
  oss << ",tooltip=\""   //
      << std::showpos    //
      << "WL=" << wl_    //
      << std::noshowpos  //
      << "\\nD=" << d_ << "\\nM=" << m_ << "\\nN=" << weight_
      << "\\nNP=" << num_parents_
      << "\\nTerm=" << static_cast<int>(terminal_type_)  //
      << std::showpos                                    //
      << "\\nBounds=" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2 << std::noshowpos  //
      << "\\n\\nThis=" << this << "\\nEdges=" << GetEdges()
      << "\\nNumEdges=" << static_cast<int>(num_edges_)
      << "\\nChild=" << GetChild() << "\\n\"";
  oss << "];" << std::endl;
}

void Node::DotEdgeString(std::ofstream& oss, bool as_opponent,
                         const LowNode* parent) const {
  oss << (parent == nullptr ? "top" : PtrToNodeName(parent)) << " -> "
      << (low_node_ ? PtrToNodeName(low_node_.get()) : PtrToNodeName(this))
      << " [";
  oss << "label=\""
      << (parent == nullptr ? "N/A" : GetMove(as_opponent).ToString(true))
      << "\\lN=" << weight_ << "\\lN_=" << GetNInFlight();
  oss << "\\l\"";
  // Set precision for tooltip.
  oss << std::fixed << std::setprecision(5);
  oss << ",labeltooltip=\""
      << "P=" << (parent == nullptr ? 0.0f : GetP())  //
      << std::showpos                                 //
      << "\\nWL= " << wl_                             //
      << std::noshowpos                               //
      << "\\nD=" << d_ << "\\nM=" << m_ << "\\nN=" << weight_
      << "\\nN_=" << GetNInFlight()
      << "\\nTerm=" << static_cast<int>(terminal_type_)  //
      << std::showpos                                    //
      << "\\nBounds=" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2 << "\\n\\nThis=" << this  //
      << std::noshowpos                                               //
      << "\\nLowNode=" << low_node_.get() << "\\nParent=" << parent
      << "\\nIndex=" << index_ << "\\nSibling=" << sibling_.get() << "\\n\"";
  oss << "];" << std::endl;
}

void Node::DotGraphString(std::ofstream& oss, bool as_opponent) const {
  oss << "strict digraph {" << std::endl;
  oss << "edge ["
      << "headport=n"
      << ",tooltip=\" \""  // Remove default tooltips from edge parts.
      << "];" << std::endl;
  oss << "node ["
      << "shape=point"    // For fake nodes.
      << ",style=filled"  // Show tooltip everywhere on the node.
      << ",fillcolor=ivory"
      << "];" << std::endl;
  oss << "ranksep=" << 4.0f * std::log10(GetN()) << std::endl;

  TreeWalk(
      this, !as_opponent,
      [&](const LowNode* low_node, bool) { low_node->DotNodeString(oss); },
      [&](const Node* node, bool as_opponent, const LowNode* parent) {
        node->DotEdgeString(oss, as_opponent, parent);
      });

  oss << "}" << std::endl;
}

bool Node::ZeroNInFlight() const {
  size_t nonzero_node_count = 0;
  TreeWalk(
      this, false, [](const LowNode*, bool) {},
      [&](const Node* node, bool, const LowNode*) {
        if (node->GetNInFlight() > 0) [[unlikely]] {
          CERR << node->DebugString() << std::endl;
          ++nonzero_node_count;
        }
      });

  if (nonzero_node_count > 0) {
    CERR << "GetNInFlight() is nonzero on " << nonzero_node_count << " nodes"
         << std::endl;
    return false;
  }

  return true;
}
#endif

void Node::SortEdges() const {
  assert(low_node_);
  low_node_->SortEdges();
}

static constexpr double wld_tolerance = 0.000001f;
static constexpr float m_tolerance = 0.000001f;

static bool WLDMInvariantsHold(double wl, double d, float m) {
  return -(1.0f + wld_tolerance) < wl && wl < (1.0f + wld_tolerance) &&  //
         -(0.0f + wld_tolerance) < d && d < (1.0f + wld_tolerance) &&    //
         -(0.0f + m_tolerance) < m &&                                    //
         std::abs(wl) + std::abs(d) < (1.0f + wld_tolerance);
}

bool Node::WLDMInvariantsHold() const {
  if (dag_classic::WLDMInvariantsHold(GetWL(), GetD(), GetM())) return true;

  std::cerr << DebugString() << std::endl;

  return false;
}

bool LowNode::WLDMInvariantsHold() const {
  if (dag_classic::WLDMInvariantsHold(GetWL(), GetD(), GetM())) return true;

  std::cerr << DebugString() << std::endl;

  return false;
}

/////////////////////////////////////////////////////////////////////////
// EdgeAndNode
/////////////////////////////////////////////////////////////////////////

std::string EdgeAndNode::DebugString() const {
  if (!edge_) return "(no edge)";
  return edge_->DebugString() + " " +
         (node_ ? node_->DebugString() : "(no node)");
}

/////////////////////////////////////////////////////////////////////////
// NodeTree
/////////////////////////////////////////////////////////////////////////

NodeTree::~NodeTree() {
  auto& ngc = NGC::Instance();
  ngc.AddToGcQueue(gamebegin_node_);
  ngc.NotifyThreadGoingSleep();
  // Start garbage collection now because we delete everything.
  ngc.Start();
}

void NodeTree::MakeMove(Move move) {
  current_head_ = current_head_->MakeSingleChild(move);
  // Ensure head is not terminal, so search can extend or visit children of
  // "terminal" positions, e.g., WDL hits, converted terminals, 3-fold draw.
  if (current_head_->IsTerminal()) current_head_->MakeNotTerminal();
  history_.Append(move);
  moves_.push_back(move);
}

void NodeTree::TrimTreeAtHead() {
  current_head_->Trim();
  // Flush the thread local destruction queue.
  NGC::Instance().NotifyThreadGoingSleep();
}

bool NodeTree::ResetToPosition(const GameState& pos) {
  if (gamebegin_node_ && (history_.Starting() != pos.startpos)) {
    // Completely different position.
    DeallocateTree();
  }

  if (!gamebegin_node_) {
    gamebegin_node_ = std::make_unique<Node>(0);
  }

  history_.Reset(pos.startpos);
  moves_.clear();

  Node* old_head = current_head_;
  current_head_ = gamebegin_node_.get();
  bool seen_old_head = (gamebegin_node_.get() == old_head);
  for (const Move m : pos.moves) {
    MakeMove(m);
    if (old_head == current_head_) seen_old_head = true;
  }

  // MakeMove guarantees that no siblings exist; but, if we didn't see the old
  // head, it means we might have a position that was an ancestor to a
  // previously searched position, which means that the current_head_ might
  // retain old n_ and q_ (etc) data, even though its old children were
  // previously trimmed; we need to reset current_head_ in that case.
  if (!seen_old_head) TrimTreeAtHead();
  NGC::Instance().NotifyThreadGoingSleep();
  return seen_old_head;
}

bool NodeTree::ResetToPosition(const std::string& starting_fen,
                               const std::vector<std::string>& moves) {
  GameState state;
  state.startpos = Position::FromFen(starting_fen);
  ChessBoard cur_board = state.startpos.GetBoard();
  state.moves.reserve(moves.size());
  for (const auto& move : moves) {
    Move m = cur_board.ParseMove(move);
    state.moves.push_back(m);
    cur_board.ApplyMove(m);
    cur_board.Mirror();
  }
  return ResetToPosition(state);
}

void NodeTree::DeallocateTree() {
  NGC::Instance().AddToGcQueue(gamebegin_node_);
  current_head_ = nullptr;
}

template <typename Types>
NodeGarbageCollector<Types>::NodeGarbageCollector()
    : gc_thread_{[this]() { GCThread(); }} {}

template <typename Types>
template <typename Ptr>
void NodeGarbageCollector<Types>::AddToGcQueue(Ptr& shared_node) {
  using Type = std::remove_cvref_t<decltype(*shared_node)>;
  std::unique_ptr<Type> node;
  if constexpr (std::is_pointer_v<Ptr>) {
    node.reset(shared_node);
  } else {
    node.reset(shared_node.release());
  }
  if (ShouldQueue(!!node)) {
    LocalWork().emplace_back(std::move(node));
  }
}

template <typename Types>
NodeGarbageCollector<Types>::~NodeGarbageCollector() {
  state_.store(Exit, std::memory_order_release);
#ifndef NO_STD_ATOMIC_WAIT
  state_.notify_all();
#else
  {
    Mutex::Lock lock(state_mutex_);
    state_signal_.notify_all();
  }
#endif
  gc_thread_.join();
}

template <typename Types>
bool NodeGarbageCollector<Types>::SetState(State& old, State desired) {
  bool rv =
      state_.compare_exchange_strong(old, desired, std::memory_order_acq_rel);
  if (rv) {
#ifndef NO_STD_ATOMIC_WAIT
    state_.notify_all();
#else
    Mutex::Lock lock(state_mutex_);
    state_signal_.notify_all();
#endif
  }
  return rv;
}

template <typename Types>
void NodeGarbageCollector<Types>::Start() {
  State s = state_.load(std::memory_order_acquire);
  do {
    if (s == Running) break;
    assert(s != Exit);
  } while (!SetState(s, Running));
}

template <typename Types>
void NodeGarbageCollector<Types>::Stop() {
  State old = state_.load(std::memory_order_acquire);
  do {
    if (old != Running && old != Waiting) break;
  } while (!SetState(old, GoToSleep));
}

template <typename Types>
void NodeGarbageCollector<Types>::Abort() {
  Stop();
}

template <typename Types>
NodeGarbageCollector<Types>::State NodeGarbageCollector<Types>::Wait() const {
  State s;
  while ((s = state_.load(std::memory_order_acquire)) != Sleeping) {
    assert(s != Exit);
#ifndef NO_STD_ATOMIC_WAIT
    state_.wait(s, std::memory_order_acquire);
#else
    Mutex::Lock lock(state_mutex_);
    state_signal_.wait(lock.get_raw(), [this, s]() { return s != state_; });
#endif
  }
  return s;
}

template <typename Types>
void NodeGarbageCollector<Types>::NotifyThreadGoingSleep() {
  if (LocalWork().empty()) {
    return;
  }
  ReleaseNodesWork<Types> new_work;
  LocalWork().swap(new_work);
}

template <typename Types>
bool NodeGarbageCollector<Types>::IsActive() const {
  auto s = state_.load(std::memory_order_acquire);
  return s == Running || s == Waiting;
}

template <typename Types>
bool NodeGarbageCollector<Types>::ShouldQueue(bool holds_node) const {
  // We don't want to queue null pointers.
  if (!holds_node) {
    return false;
  }

  // If state is exit, it means thread local queues have been destroyed.
  State s = state_.load(std::memory_order_acquire);
  if (s == Exit) {
    return false;
  }

  // We directly free the node, if queue is running and we are in the GC
  // thread. All other queue request should be pushed to the thread local
  // batch.
  return s != Running || !LocalWork().IsWorker();
}

template <typename Types>
void NodeGarbageCollector<Types>::GCThread() {
  auto& shared_work = LocalWork(true);
  assert(shared_work.IsWorker());
  State s;
  while ((s = state_.load(std::memory_order_acquire)) != Exit) {
    if (s == GoToSleep) {
      // Signal other threads that we have stopped destruction work.
      if (SetState(s, Sleeping)) {
        s = Sleeping;
      } else {
        continue;
      }
    }
    if (s == Sleeping || s == Waiting) {
#ifndef NO_STD_ATOMIC_WAIT
      state_.wait(s, std::memory_order_acquire);
#else
      Mutex::Lock lock(state_mutex_);
      state_signal_.wait(lock.get_raw(),
                         [this]() { return Sleeping != state_; });
#endif
      if (!shared_work.empty()) {
        // Check for early exit from previous free. The work can be freed
        // before the batch is full.
        ReleaseNodesWork<Types> new_work(true);
        new_work.swap(shared_work);
      }
      continue;
    }

    assert(s == Running);

    LCTRACE_FUNCTION_SCOPE;

    bool empty = true;
    std::vector<Types> nodes;
    {
      SpinMutex::Lock lock(mutex_);
      if (!released_nodes_.empty()) {
        empty = false;
        nodes = std::move(released_nodes_.front());
        released_nodes_.pop_front();
      }
    }

    if (!empty) {
      LOGFILE << "Garbage collection starting.";
    }

    // Free nodes one by one. LowNode destructor calls AddToGcQueue which
    // allows recursive destruction terminate before freeing a whole branch.
    while (!nodes.empty()) {
      if (!IsActive()) {
        break;
      }
      nodes.pop_back();
    }

    if (!empty) {
      LOGFILE << "Garbage collection ending.";
    }

    // Go to sleep if empty or search stopped.
    if (empty || !IsActive()) {
      // Lock is requrired to avoid race between other thread queueing work
      // and calling Start().
      SpinMutex::Lock lock(mutex_);
      // There wasn't enough time to free all nodes. They must go back to the
      // list.
      if (!nodes.empty()) {
        released_nodes_.emplace_front(std::move(nodes));
      }

      // Going to sleep if the queue is empty.
      if (released_nodes_.empty()) {
        State old = Running;
        SetState(old, Waiting);
      }
    }
  }
}

template <typename Types>
ReleaseNodesWork<Types>::ReleaseNodesWork(bool gc_thread)
    : is_gc_thread_(gc_thread) {
  released_nodes_.reserve(kCapacity);
}

template <typename Types>
bool ReleaseNodesWork<Types>::IsWorker() const {
  return is_gc_thread_;
}

template <typename Types>
template <typename T>
void ReleaseNodesWork<Types>::emplace_back(std::unique_ptr<T>&& node) {
  if (!node) return;
  released_nodes_.emplace_back(Types{std::move(node)});
  if (released_nodes_.size() == kCapacity) {
    ReleaseNodesWork new_work(is_gc_thread_);
    swap(new_work);
  }
}

template <typename Types>
bool ReleaseNodesWork<Types>::empty() const {
  return released_nodes_.empty();
}

template <typename Types>
void ReleaseNodesWork<Types>::swap(ReleaseNodesWork& other) {
  assert(IsWorker() == other.IsWorker());
  std::swap(released_nodes_, other.released_nodes_);
}

template <typename Types>
ReleaseNodesWork<Types>::~ReleaseNodesWork() {
  Submit();
}

template <typename Types>
void ReleaseNodesWork<Types>::Submit() {
  if (released_nodes_.empty()) {
    return;
  }
  LCTRACE_FUNCTION_SCOPE;
  auto& worker = NodeGarbageCollector<Types>::Instance();
  SpinMutex::Lock lock(worker.mutex_);
  // If this is worker, we have oldest nodes. Keep them at front of the queue.
  if (IsWorker()) {
    worker.released_nodes_.emplace_front(std::move(released_nodes_));
  } else {
    worker.released_nodes_.emplace_back(std::move(released_nodes_));
    if (worker.released_nodes_.size() > worker.kQueueWakeupThreshold) {
      // Notify the worker thread about new work if it is waiting.
      auto old = NodeGarbageCollector<Types>::Waiting;
      worker.SetState(old, NodeGarbageCollector<Types>::Running);
    }
  }
}

template class NodeGarbageCollector<NGCTypes>;

}  // namespace dag_classic
}  // namespace lczero
