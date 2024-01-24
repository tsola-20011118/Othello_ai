import pyxel
import numpy as np
from multiprocessing import Process, Pipe
from tqdm import tqdm

class QLearningOthello:
	def __init__(self, board_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
		# ゲームボードのサイズと学習パラメータの設定
		self.board_size = board_size
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.exploration_rate = exploration_rate

		# Q-tableのサイズを縮小
		self.q_table_size = 5000  # 適切なサイズに調整
		self.q_table = np.zeros((self.q_table_size, board_size ** 2))

	def state_to_index(self, state):
		"""
		ゲームボードの状態をQ-tableのインデックスに変換する
		"""
		index = 0
		multiplier = 1
		for value in state.flatten():
			index += value * multiplier
			multiplier *= 3  # 3進数として扱う
		return index % self.q_table_size

	def get_q_values(self, state):
		return self.q_table[self.state_to_index(state)]

	def choose_action(self, state):
		# ε-greedy方針に基づき行動を選択
		if np.random.rand() < self.exploration_rate:
			return np.random.randint(0, self.board_size ** 2)
		else:
			q_values = self.get_q_values(state)
			if not np.any(q_values):
				return np.random.randint(0, self.board_size ** 2)
			return np.argmax(q_values)

	def update_q_value(self, state, action, reward, next_state):
		# Q-valueの更新
		current_q = self.get_q_values(state)[action]
		max_future_q = max(self.get_q_values(next_state), default=0)
		new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
		self.q_table[self.state_to_index(state)][action] = new_q

	def train(self, episodes=10, progress_pipe=None):
		# Q-learningによるエージェントのトレーニング
		for episode in range(episodes):
			game = OthelloGame(agent=self)

			while not game.is_game_over():
				state = np.array(game.board)
				action = self.choose_action(state)

				prev_score = self.evaluate_board(state)
				game.place_piece(action // game.board_size, action % game.board_size)

				current_score = self.evaluate_board(np.array(game.board))
				reward = current_score - prev_score

				next_state = np.array(game.board)
				self.update_q_value(state, action, reward, next_state)

				if game.is_game_over():
					final_score = self.evaluate_board(np.array(game.board))
					reward = final_score - current_score
					self.update_q_value(next_state, action, reward, next_state)

			# 進捗を送信
			if progress_pipe:
				progress_pipe.send(1)

	def evaluate_board(self, board):
		# ゲームボードの評価
		black_score = np.sum(np.array(board) == 1)
		white_score = np.sum(np.array(board) == 2)

		if black_score > white_score:
			return 1
		elif black_score < white_score:
			return -1
		else:
			return 0

	def play(self, human_player=True):
		# ゲームの実行（pyxelを使用）
		pyxel.init(120, 120, fps=10)
		game = OthelloGame(agent=self, human_player=human_player)
		game.battle = True

		def update():
			game.update()

		def draw():
			pyxel.cls(2)
			for i in range(self.board_size):
				for j in range(self.board_size):
					pyxel.circ(i * 20 + 10, j * 20 + 10, 8, 3)
			game.draw()
			if game.is_game_over():
				game.draw_game_over()
			pyxel.rect(pyxel.mouse_x, pyxel.mouse_y, 1, 1, 0)

		pyxel.run(update, draw)

		pyxel.quit()

	def make_move(self, state):
		return self.choose_action(state)

class OthelloGame:
	def __init__(self, agent, human_player=True):
		# オセロゲームの初期化
		self.board_size = 6
		self.board = [[0] * self.board_size for _ in range(self.board_size)]
		self.current_player = 1
		self.initialize_board()
		self.agent = agent
		self.human_player = human_player
		self.battle = False

	def initialize_board(self):
		# ゲームボードの初期化
		mid = self.board_size // 2
		self.board[mid - 1][mid - 1] = 2
		self.board[mid][mid] = 2
		self.board[mid - 1][mid] = 1
		self.board[mid][mid - 1] = 1

	def draw_board(self):
		# ゲームボードの描画
		for i in range(self.board_size):
			for j in range(self.board_size):
				if self.board[i][j] == 1:
					pyxel.circ(i * 20 + 10, j * 20 + 10, 8, 0)
				elif self.board[i][j] == 2:
					pyxel.circ(i * 20 + 10, j * 20 + 10, 8, 7)

	def place_piece(self, x, y):
		# プレイヤーの駒を配置
		if self.board[x][y] == 0 and self.is_valid_move(x, y):
			self.board[x][y] = self.current_player
			self.flip_pieces(x, y)
			self.switch_player()
			if self.battle & self.human_player == False:
				self.human_player = not self.human_player

	def is_valid_move(self, x, y):
		# すでに駒が置かれている場合は置けない
		if self.board[x][y] != 0:
			return False

		# 駒を配置できるか判定
		directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

		for dx, dy in directions:
			nx, ny = x + dx, y + dy
			while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][ny] == 3 - self.current_player:
				nx, ny = nx + dx, ny + dy
				if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][ny] == self.current_player:
					return True
		return False

	def flip_pieces(self, x, y):
		# 駒を裏返す
		directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

		for dx, dy in directions:
			nx, ny = x + dx, y + dy
			while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][ny] == 3 - self.current_player:
				nx, ny = nx + dx, ny + dy
				if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][ny] == self.current_player:
					nx, ny = x + dx, y + dy
					while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][ny] == 3 - self.current_player:
						self.board[nx][ny] = self.current_player
						nx, ny = nx + dx, ny + dy

	def switch_player(self):
		# プレイヤーの切り替え
		self.current_player = 3 - self.current_player

	def is_game_over(self):
		# ゲーム終了判定
		for i in range(self.board_size):
			for j in range(self.board_size):
				# 未配置のマスで、かつそのマスに駒を置ける場合
				if self.board[i][j] == 0 and self.is_valid_move(i, j):
					# ゲームはまだ終了していない
					return False
		# 上記条件を満たすマスが存在しない場合、ゲームは終了している
		return True


	def evaluate_board(self, board):
		# ゲームボードの評価
		black_score = np.sum(np.array(board) == 1)
		white_score = np.sum(np.array(board) == 2)

		if black_score > white_score:
			return 1
		elif black_score < white_score:
			return -1
		else:
			return 0

	def draw_game_over(self):
		# ゲーム終了時のメッセージ描画
		winner = self.get_winner()
		if winner == 0:
			pyxel.text(50, 70, "Draw!", 8)
		else:
			if winner == 1:
				pyxel.text(50, 70, f"Player Wins!", 8)
			else:
				pyxel.text(50, 70, f"AI Wins!", 8)

	def update(self):
		# ゲームの更新
		if self.human_player and self.current_player == 1:
			if pyxel.mouse_x > 0 and pyxel.mouse_x < 120 and pyxel.mouse_y > 0 and pyxel.mouse_y < 120 and pyxel.btnp(pyxel.KEY_SPACE, 1, 1):
				x, y = pyxel.mouse_x // 20, pyxel.mouse_y // 20
				if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == 0:
					self.place_piece(x, y)
					self.human_player = not self.human_player
		else:
			if not self.is_game_over():
				state = np.array(self.board)
				action = self.agent.make_move(state)
				x, y = action // self.board_size, action % self.board_size
				self.place_piece(x, y)

	def draw(self):
		pyxel.cls(2)
		for i in range(self.board_size):
			for j in range(self.board_size):
				# 自分の手番の場合、駒を置けるますの色を変更
				if self.human_player and self.current_player == 1 and self.is_valid_move(i, j):
					pyxel.rect(i * 20, j * 20, 20, 20, 9)  # 9 は別の色に変更可能
				pyxel.circ(i * 20 + 10, j * 20 + 10, 8, 3)
		self.draw_board()
		if self.is_game_over():
			self.draw_game_over()
		pyxel.rect(pyxel.mouse_x, pyxel.mouse_y, 1, 1, 0)

	def get_winner(self):
		# 勝者の取得
		black_score = np.sum(np.array(self.board) == 1)
		white_score = np.sum(np.array(self.board) == 2)

		if black_score > white_score:
			return 1
		elif black_score < white_score:
			return 2
		else:
			return 0

def train_agent(agent, episodes, progress_pipe):
    # エージェントのトレーニング
    agent.train(episodes=episodes, progress_pipe=progress_pipe)

if __name__ == "__main__":
	q_learning_agent = QLearningOthello(board_size=6)

	# 進捗表示用のパイプを作成
	progress_pipe, child_conn = Pipe()

	# トレーニング用プロセスを起動
	training_process = Process(target=train_agent, args=(q_learning_agent, 1000, child_conn))
	training_process.start()

	# 学習の進捗を表示
	with tqdm(total=1000, desc="Training", position=0, leave=True) as pbar:
		while training_process.is_alive():
			if progress_pipe.poll():
				pbar.update(progress_pipe.recv())

	# トレーニング用プロセスの終了を待機
	training_process.join()
	print("Training Done")

	# 学習したエージェントと対戦
	q_learning_agent.play(human_player=True)
