from __future__ import annotations
import math
import random
import threading
from player import Player
from board import HexBoard

class UnionFind:
    """
    Disjoint Set Union (DSU) with lazy initialization using HashMap.
    Only allocates entries for elements that are actually used.
    """

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def _ensure_exists(self, x: int) -> None:
        """Lazily initialize an element if not present."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        """Find with path compression."""
        self._ensure_exists(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are in the same set."""
        return self.find(x) == self.find(y)


class HexBoardDSU:
    """
    Hex board state with DSU for efficient win checking.
    Uses virtual nodes for board edges to check connectivity.

    For a board of size n:
    - Cells are indexed 0 to n*n - 1
    - Virtual nodes for Player 1 (left/right): n*n and n*n + 1
    - Virtual nodes for Player 2 (top/bottom): n*n + 2 and n*n + 3
    """

    EVEN_OFFSETS = [(0, -1), (0, 1), (-1, -1), (-1, 0), (1, -1), (1, 0)]
    ODD_OFFSETS = [(0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)]

    def __init__(self, size: int, board: list = []):
        self.size = size
        self.board = [row[:] for row in board] if board else [[0] * size for _ in range(size)]

        # Create DSU with lazy initialization
        n = size * size
        self.dsu_p1 = UnionFind()
        self.dsu_p2 = UnionFind()

        self.left_virtual = n       # Player 1's left edge
        self.right_virtual = n + 1  # Player 1's right edge
        self.top_virtual = n        # Player 2's top edge
        self.bottom_virtual = n + 1 # Player 2's bottom edge

        # Initialize DSU with existing pieces
        for r in range(size):
            for c in range(size):
                if self.board[r][c] != 0:
                    self._connect_cell(r, c, self.board[r][c])

    def _cell_index(self, r: int, c: int) -> int:
        """Convert (row, col) to linear index."""
        return r * self.size + c

    def _get_neighbors(self, row: int, col: int) -> list:
        """Returns valid neighbor coordinates using even-r adjacency."""
        offsets = self.EVEN_OFFSETS if row % 2 == 0 else self.ODD_OFFSETS
        neighbors = []
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append((nr, nc))
        return neighbors

    def _connect_cell(self, r: int, c: int, player_id: int) -> None:
        """Connect a cell to its neighbors and virtual edges in the DSU."""
        cell_idx = self._cell_index(r, c)

        if player_id == 1:
            dsu = self.dsu_p1
            # Connect to left virtual if on left edge
            if c == 0:
                dsu.union(cell_idx, self.left_virtual)
            # Connect to right virtual if on right edge
            if c == self.size - 1:
                dsu.union(cell_idx, self.right_virtual)
        else:
            dsu = self.dsu_p2
            # Connect to top virtual if on top edge
            if r == 0:
                dsu.union(cell_idx, self.top_virtual)
            # Connect to bottom virtual if on bottom edge
            if r == self.size - 1:
                dsu.union(cell_idx, self.bottom_virtual)

        # Connect to neighboring cells of the same player
        for nr, nc in self._get_neighbors(r, c):
            if self.board[nr][nc] == player_id:
                neighbor_idx = self._cell_index(nr, nc)
                dsu.union(cell_idx, neighbor_idx)

    def place_piece(self, r: int, c: int, player_id: int) -> bool:
        """Place a piece and update DSU."""
        if self.board[r][c] != 0:
            return False
        self.board[r][c] = player_id
        self._connect_cell(r, c, player_id)
        return True

    def check_winner(self, player_id: int) -> bool:
        """Check if player has won using DSU connectivity."""
        if player_id == 1:
            return self.dsu_p1.connected(self.left_virtual, self.right_virtual)
        else:
            return self.dsu_p2.connected(self.top_virtual, self.bottom_virtual)

    def get_empty_cells(self) -> list:
        """Return list of empty cell coordinates."""
        empty = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    empty.append((r, c))
        return empty

    def clone(self) -> "HexBoardDSU":
        """Create a deep copy of this board state."""
        return HexBoardDSU(self.size, self.board)


class MCTSNode:
    """A node in the Monte Carlo Tree Search tree."""

    def __init__(self, board: HexBoardDSU, player_to_move: int, parent=None, move=None):
        self.board = board
        self.player_to_move = player_to_move
        self.parent: MCTSNode|None = parent
        self.move = move  # The move that led to this node

        self.wins = 0
        self.visits = 0
        self.children = []
        self.untried_moves = board.get_empty_cells()

    def is_fully_expanded(self) -> bool:
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (game over)."""
        return (self.board.check_winner(1) or
                self.board.check_winner(2) or
                len(self.board.get_empty_cells()) == 0)

    def ucb1(self, exploration_constant: float = 1.41) -> float:
        """Calculate UCB1 value for node selection."""
        if self.visits == 0 or self.parent is None:
            return float('inf')

        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self) -> "MCTSNode":
        """Select child with highest UCB1 value."""
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self) -> "MCTSNode":
        """Expand by adding a new child node for an untried move."""
        move = self.untried_moves.pop()
        new_board = self.board.clone()
        new_board.place_piece(move[0], move[1], self.player_to_move)

        next_player = 3 - self.player_to_move
        child = MCTSNode(new_board, next_player, parent=self, move=move)
        self.children.append(child)
        return child

    def backpropagate(self, winner: int) -> None:
        """Backpropagate the result up the tree."""
        node = self
        while node is not None:
            node.visits += 1
            # Win is counted from the perspective of the player who just moved
            # (the parent's player_to_move)
            if node.parent is not None:
                player_who_moved = node.parent.player_to_move
                if winner == player_who_moved:
                    node.wins += 1
            node = node.parent


class MCTSPlayer(Player):
    """
    Monte Carlo Tree Search player for Hex.

    MCTS consists of four phases repeated within a time limit:
    1. Selection: Traverse tree using UCB1 until reaching unexpanded node
    2. Expansion: Add a new child node for an untried move
    3. Simulation: Play randomly until game ends
    4. Backpropagation: Update statistics up the tree
    """

    def __init__(self, player_id: int, time_limit: float = 0.5):
        super().__init__(player_id)
        self.time_limit = time_limit


    def play(self, board) -> tuple:
        """Select the best move using MCTS with a strict hardware timer."""
        dsu_board = HexBoardDSU(board.size, board.board)
        root = MCTSNode(dsu_board, self.player_id)

        legal_moves = dsu_board.get_empty_cells()
        if not legal_moves:
            raise RuntimeError("No legal moves available")
        
        # Store state in a dictionary so the worker thread can mutate it
        state = {
            "best_move": random.choice(legal_moves),
            "stop_event": threading.Event()
        }

        def mcts_worker():
            while not state["stop_event"].is_set():
                node = self._select(root)
                winner = self._simulate(node)
                node.backpropagate(winner)

                if root.children:
                    best_child = max(root.children, key=lambda c: c.visits)
                    state["best_move"] = best_child.move

        # Start MCTS in the background
        worker = threading.Thread(target=mcts_worker)
        worker.daemon = True 
        worker.start()

        # The main thread stops here and waits EXACTLY for the time_limit
        worker.join(timeout=self.time_limit)

        # Signal the background thread to safely spin down
        state["stop_event"].set() 

        return state["best_move"]

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: descend tree using UCB1 until finding expandable node."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            node = node.select_child()
        return node

    def _simulate(self, node: MCTSNode) -> int:
        """
        Simulation phase: play random moves until game ends.
        Returns the winner (1 or 2) or 0 for draw.
        """
        board = node.board.clone()
        current_player = node.player_to_move

        # Check if game already ended
        if board.check_winner(1):
            return 1
        if board.check_winner(2):
            return 2

        empty_cells = board.get_empty_cells()
        random.shuffle(empty_cells)

        # Play out randomly
        for move in empty_cells:
            board.place_piece(move[0], move[1], current_player)

            if board.check_winner(current_player):
                return current_player

            current_player = 3 - current_player

        return 0  # Draw (shouldn't happen in Hex)
