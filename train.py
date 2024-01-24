import pyxel
import numpy as np
from multiprocessing import Process, Pipe
from tqdm import tqdm
import matplotlib.pyplot as plt


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
		self.score = 0

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
			# # 進捗を送信
			# if progress_pipe:
			# 	progress_pipe.send(1)
		

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
			if pyxel.mouse_x > 0 and pyxel.mouse_x < 120 and pyxel.mouse_y > 0 and pyxel.mouse_y < 120 and pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT):
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
		total_pieces = 0  # 合計数を保持する変数を初期化
		for i in range(self.board_size):
			for j in range(self.board_size):
				# 自分の手番の場合、駒を置けるますの色を変更
				if self.human_player and self.current_player == 1 and self.is_valid_move(i, j):
					if i * 20 <= pyxel.mouse_x <= (i + 1) * 20 and j * 20 <= pyxel.mouse_y <= (j + 1) * 20:
						pyxel.rect(i * 20, j * 20, 20, 20, 8)  # マウスが当たっている場合、背景色を変更
					else:
						pyxel.rect(i * 20, j * 20, 20, 20, 9)  # 9 は別の色に変更可能
				pyxel.circ(i * 20 + 10, j * 20 + 10, 8, 3)  # マスの中央に円を描画
		self.draw_board()
		if self.is_game_over():
			self.draw_game_over()

		# 合計数を表示
		pyxel.text(30, 120, f"YOU  {np.sum(np.array(self.board) == 1)} : {np.sum(np.array(self.board) == 2)}  AI", 0)
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

	def reset_game(self):
		# ゲームの初期化
		self.board = [[0] * self.board_size for _ in range(self.board_size)]
		self.initialize_board()
		self.current_player = 1

	def play_against(self, agent1, agent2):
		while not self.is_game_over():
			if self.current_player == 1:
				# Player 1 (agent1)の手番
				state = np.array(self.board)
				action = agent1.make_move(state)
				x, y = action // self.board_size, action % self.board_size
				self.place_piece(x, y)
			else:
				# Player 2 (agent2)の手番
				state = np.array(self.board)
				action = agent2.make_move(state)
				x, y = action // self.board_size, action % self.board_size
				self.place_piece(x, y)
		if np.sum(np.array(game.board) == 1) >= 18:
			agent1.score += 1
		else:
			agent2.score += 1

def train_agent(agent, episodes, progress_pipe):
	# エージェントのトレーニング
	agent.train(episodes=episodes, progress_pipe=progress_pipe)

def plot_win_rates(win_rates, win_rate, exploration2, episode2):
	plt.clf() 

	# 棒グラフの表示
	plt.bar([str((i+1) * 10) for i in range(len(win_rates))], win_rates)

	# グラフのタイトルと軸ラベル
	plt.xlabel("Battle Number") #対戦回数
	plt.ylabel("Winning Probability") #勝率
	plt.title("Winning Probability over Battles") # 対戦回数に対する勝率の推移

	# グラフの表示
	# plt.show()
	# y軸の範囲を0から1に設定
	plt.ylim(0, 1)
	plt.text(0, 0.9, "Win rate:" + str(round(win_rate, 4)*100) + "%", fontsize=12, color='red')
	plt.text(0, 0.85, "Training exploration:" + exploration2, fontsize=12, color='black')
	plt.text(0, 0.8, "Training episode:" + episode2, fontsize=12, color='black')
	plt.savefig("winning_probability_plot.png")

if __name__ == "__main__":
	# 人間と対戦用
	if input('人間と対戦しますか: y or n : ') == "y":
		agent_num = 1000
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


		# ゲームの初期化
		pyxel.init(120, 140, fps=10)
		game = OthelloGame(agent=q_learning_agent, human_player=True)
		game.battle = True

		def update():
			if game.is_game_over():
				if pyxel.btnp(pyxel.KEY_SPACE, 1, 1):
					if np.sum(np.array(game.board) == 1) >= 18:
						q_learning_agent.score += 1
					# ゲームが終了したら初期化
					game.reset_game()
			else:
				game.update()

		def draw():
			game.draw()
			pyxel.text(5, 130, f"Your Scores: {q_learning_agent.score}", 0)
		pyxel.run(update, draw)

		#エージェント同士が戦う

	#発表用
	elif input('学習回数の実験を行いますか: y or n : ') == "y":
		episodes1 = 100
		episodes2 = 10
		q_learning_agent1 = QLearningOthello(board_size=6)
		q_learning_agent2 = QLearningOthello(board_size=6)

		progress_pipe1, child_conn1 = Pipe()
		progress_pipe2, child_conn2 = Pipe()

		training_process1 = Process(target=train_agent, args=(q_learning_agent1, episodes1, child_conn1))
		training_process2 = Process(target=train_agent, args=(q_learning_agent2, episodes2, child_conn2))

		training_process1.start()
		training_process2.start()

		with tqdm(total=episodes1, desc="Training Agent 1", position=0, leave=True) as pbar1, \
				tqdm(total=episodes2, desc="Training Agent 2", position=1, leave=True) as pbar2:

			while training_process1.is_alive() or training_process2.is_alive():
				if progress_pipe2.poll():
					pbar2.update(progress_pipe2.recv())
				if progress_pipe1.poll():
					pbar1.update(progress_pipe1.recv())

		training_process1.join()
		training_process2.join()
		print("Training Done")

		trials = 500 + 1
		exploration2 = 0.1
		while True:
			# 対戦回数ごとに勝率を保存するリスト
			win_rates = []
			for j in range(trials):
				# エージェント同士の対戦
				game = OthelloGame(agent=q_learning_agent1, human_player=False)
				game.battle = True
				game.play_against(q_learning_agent1, q_learning_agent2)
				if j % 100 == 0 and j != 0 :
					# 勝率を計算してリストに追加
					win_rate = q_learning_agent2.score / (q_learning_agent1.score + q_learning_agent2.score)
					win_rates.append(win_rate)
					episodes2 += 100
					# エージェント2の学習
					training_process2 = Process(target=train_agent, args=(q_learning_agent2, episodes2 - 10, child_conn2))
					training_process2.daemon = True
					training_process2.start()
					training_process2.join()
					q_learning_agent1.score = 0
					q_learning_agent2.score = 0
					print(str(episodes2 - 10))
			# 勝率の推移をグラフにプロット
			plot_win_rates(win_rates, win_rate, str(exploration2), str(episodes2))
			# 対戦結果の表示
			print(f"Agent 1 Scores: {q_learning_agent1.score}")
			print(f"Agent 2 Scores: {q_learning_agent2.score}")
			# 勝率の推移をグラフにプロット
			plot_win_rates(win_rates, win_rate, str(exploration2), str(episodes2))
			print("勝率の推移を出力しました")
			e_change = "n"
			n_change = "n"
			e_change = input('エージェントのεを変更しますか: y or n : ')
			# 学習ε変更
			if e_change == "y":
				exploration2 = input('εを幾つにしますか(半角数字で入力): ')
				while not(0.0 <= float(exploration2) <= 1.0):
					exploration2 = input('εを幾つにしますか(半角数字で入力)(0~1で入力してください): ')
				# エージェント2の学習
				q_learning_agent2 = QLearningOthello(board_size=6, exploration_rate=float(exploration2))
			n_change = input('エージェントの試行回数を変更しますか: y or n : ')
	
	#エージェント同士が戦う
	else:
		episodes1 = 10000
		episodes2 = 10000
		q_learning_agent1 = QLearningOthello(board_size=6)
		q_learning_agent2 = QLearningOthello(board_size=6)

		progress_pipe1, child_conn1 = Pipe()
		progress_pipe2, child_conn2 = Pipe()

		training_process1 = Process(target=train_agent, args=(q_learning_agent1, episodes1, child_conn1))
		training_process2 = Process(target=train_agent, args=(q_learning_agent2, episodes2, child_conn2))

		training_process1.start()
		training_process2.start()

		with tqdm(total=episodes1, desc="Training Agent 1", position=0, leave=True) as pbar1, \
				tqdm(total=episodes2, desc="Training Agent 2", position=1, leave=True) as pbar2:

			while training_process1.is_alive() or training_process2.is_alive():
				if progress_pipe2.poll():
					pbar2.update(progress_pipe2.recv())
				if progress_pipe1.poll():
					pbar1.update(progress_pipe1.recv())

		training_process1.join()
		training_process2.join()
		print("Training Done")

		battle_num = 100
		trials = 10000
		exploration2 = 0.1
		while True:
			# 対戦回数ごとに勝率を保存するリスト
			win_rates = []
			for j in range(trials):
				if(j % 1 == 0):print("j:      " + str(j))
				for i in range(battle_num):
					if(i % 100 == 0):print("i: " + str(i))
					# エージェント同士の対戦
					game = OthelloGame(agent=q_learning_agent1, human_player=False)
					game.battle = True
					game.play_against(q_learning_agent1, q_learning_agent2)
				if j % 10 == 0:
					# 勝率を計算してリストに追加
					win_rate = q_learning_agent2.score / (q_learning_agent1.score + q_learning_agent2.score)
					win_rates.append(win_rate)
			# 対戦結果の表示
			print(f"Agent 1 Scores: {q_learning_agent1.score}")
			print(f"Agent 2 Scores: {q_learning_agent2.score}")
			# 勝率の推移をグラフにプロット
			plot_win_rates(win_rates, win_rate, str(exploration2), str(episodes2))
			print("勝率の推移を出力しました")
			e_change = "n"
			n_change = "n"
			e_change = input('エージェントのεを変更しますか: y or n : ')
			# 学習ε変更
			if e_change == "y":
				exploration2 = input('εを幾つにしますか(半角数字で入力): ')
				while not(0.0 <= float(exploration2) <= 1.0):
					exploration2 = input('εを幾つにしますか(半角数字で入力)(0~1で入力してください): ')
				# エージェント2の学習
				q_learning_agent2 = QLearningOthello(board_size=6, exploration_rate=float(exploration2))
			n_change = input('エージェントの試行回数を変更しますか: y or n : ')
			# 学習回数変更
			if n_change == "y":
				episodes2 = input('何回学習しますか(半角数字で入力): ')
			if e_change == "y" or n_change == "y":
				q_learning_agent1.score = 0
				q_learning_agent2.score = 0
				# エージェント2の学習
				training_process2 = Process(target=train_agent, args=(q_learning_agent2, int(episodes2), child_conn2))
				training_process2.start()
				print("学習中...")
				training_process2.join()
	
