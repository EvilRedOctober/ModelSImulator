from abc import abstractmethod, ABC
from random import random, randint, choice, shuffle
from collections import deque
from heapq import heappush, heappop
from math import hypot

import numpy as np


def talon(prob: float) -> bool:
    return prob > random()


# Basic cell type
class Special_cell(ABC):
    """Basic type for special  cell"""

    def __init__(self, x, y, *args, **kwargs):
        self.x = x
        self.y = y

    def coords(self):
        return self.x, self.y

    def move(self, x, y, *args, **kwargs):
        self.x = x
        self.y = y


# Basic model type
class Abstract_model(ABC):
    """Basic class for all models. Have abstracts methods"""
    MODEL_TEXT = ''
    COLOR_LIST = ()
    PARAMETERS = []

    def __init__(self, size: int = 10, *args, **kwargs):
        self.size = max(5, size)
        self._field = np.array(0)
        self._field_prev = np.array(0)
        self._cells = []

    @abstractmethod
    def step(self) -> list:
        """Make an one step over modelling"""
        pass

    @abstractmethod
    def start(self, *args, **kwargs) -> list:
        """Set initial conditions"""
        pass

    def is_ended(self) -> bool:
        """Returns True if there is no any changes"""
        return (self._field_prev == self._field).all()

    def get_color(self, i: int, j: int) -> str:
        """Takes the cords and return color RGB in
        string view like #HHHHHH, where H is hex digit"""
        return self.COLOR_LIST[self._field[i][j]]


# 0
class Ringworm(Abstract_model):
    """A simulation of ringworm infection"""
    MODEL_TEXT = "Моделирование инфекции стригущего лишая. В начале моделирования заражена " \
                 "центральная клетка. Каждый ход зараженная клетка с определенной вероятностью заражает " \
                 "любую соседнюю клетку. По прошествию 6 ходов зараженные клетки приобретают " \
                 "иммунитет, который длится 4 хода, после чего клетка вновь становится здоровой."
    COLOR_LIST = ('#00FF00',
                  '#3fff00', '#7eff00', '#bdff00', '#ffff00',
                  '#ffd200', '#ffa800', '#ff7e00', '#ff5400', '#ff2a00', '#ff0000')
    PARAMETERS = {'n': {'value': 1, 'min': 1, 'max': 61, 'spin_type': 'QSpinBox',
                        'name_rus': 'Число первых зараженных клеток'},
                  'prob': {'value': 0.4, 'min': 0.1, 'max': 1, 'spin_type': 'QDoubleSpinBox',
                           'name_rus': 'Вероятность заражения'}}

    def __init__(self, size: int = 10, n: int = 1, prob: float = 0.4):
        super().__init__(size)
        self.n = max(1, n)
        self.prob = prob

    def start(self):
        """The n is the number of first infected cells"""
        self._field = np.zeros((self.size, self.size), dtype='int8')
        self._field_prev = np.zeros((self.size, self.size), dtype='int8')
        # First infected cell in the center of field
        self._field[self.size // 2][self.size // 2] = 10
        # All others in random positions (there could be less than n)
        for _ in range(self.n - 1):
            i, j = randint(0, self.size - 1), randint(0, self.size - 1)
            self._field[i][j] = 10
        return [(self.size*self.size-1, 'Здоровые клетки'), (0, 'Иммунные клетки'), (1, 'Зараженные клетки')]

    def step(self):
        self._field_prev = self._field.copy()
        for i in range(self.size):
            for j in range(self.size):
                # Every infested cell exists 10 - 4 = 6 turns
                if self._field_prev[i][j] > 4:
                    # Any possible move. The edging cell are not connecting with the other side
                    coords = ((i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                              (i, j - 1), (i, j + 1),
                              (i + 1, j - 1), (i + 1, j), (i + 1, j + 1))
                    possible_coords = (coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size))
                    for coord in possible_coords:
                        # There is chance that the healthy cell will be infected
                        if self._field_prev[coord] == 0 and talon(self.prob):
                            # After 6 turns infected cell will turn into immune cell,
                            # that can't be infected, after 4 turns immune cell will lost immunity
                            self._field[coord] = 10
                # Infection is slowly go out and immunity too
                if self._field_prev[i][j]:
                    self._field[i][j] -= 1
        healthy = len(self._field[self._field == 0])
        immune = len(self._field[np.logical_and(self._field < 5, self._field > 0)])
        infested = len(self._field[self._field > 4])
        return [healthy, immune, infested]


# 1
class Wolf_Island(Abstract_model):
    """In this island there is wolves and rabbits"""
    MODEL_TEXT = "Экологическа модель острова с волками, волчицами и кроликами. Имеется по несколько " \
                 "представителей каждого вида. Кролики довольно глупы. Они перемещаются в случайном " \
                 "направлении или остаются на месте с одинаковой вероятностью. Каждый ход с определенной " \
                 "вероятностью кролик может превратиться в двух. Волчица перемещается случайно " \
                 "пока рядом не окажется кролик. Оказавшись вместе с ним в одной клетке, она съедает " \
                 "кролика и получает 10 очков, инчае теряет 1 очко. Если у нее останется 0 очков, то " \
                 "она умирает. Волк аналогичен волчице, но в случае когда рядом нет кролика, но есть волчица, " \
                 "он погонится за ней. Оказавшись в одной клетке, они дают потомство случайного пола."
    COLOR_LIST = ('#c8ff96', '#ffc832', '#c880ff', '#0096ff')
    PARAMETERS = {'r_prob': {'value': 0.2, 'min': 0.1, 'max': 0.5, 'spin_type': 'QDoubleSpinBox',
                             'name_rus': 'Процент кроликов'},
                  'w_prob': {'value': 0.05, 'min': 0.01, 'max': 0.1, 'spin_type': 'QDoubleSpinBox',
                             'name_rus': 'Процент волков и волчиц'},
                  's_prob': {'value': 0.5, 'min': 0, 'max': 1, 'spin_type': 'QDoubleSpinBox',
                             'name_rus': 'Процент самок среди волков'}}
    # 0 - empty, 1 - rabbit, 2 - she-wolf, 3 - wolf

    class Rabbit(Special_cell):
        pass

    class Wolf(Special_cell):
        def __init__(self, x, y, life):
            super().__init__(x, y)
            self.life = life

        def move(self, x, y, *args, **kwargs):
            super().move(x, y)
            self.life -= 1

    class She_Wolf(Wolf):
        def __init__(self, x, y, life):
            super().__init__(x, y, life)
            self.cooldown = 5

        def move(self, x, y, *args, **kwargs):
            super().move(x, y)
            if self.cooldown > 0:
                self.cooldown -= 1

    def __init__(self, size: int = 10, r_prob: float = 0.2, w_prob: float = 0.1, s_prob: float = 0.5):
        super().__init__(size)
        self.r_prob = r_prob
        self.w_prob = w_prob
        self.s_prob = s_prob
        self._rabbits = []
        self._she_wolves = []
        self._wolves = []

    def start(self):
        # Random animals at the island
        self._field = np.zeros((self.size, self.size), dtype='int8')
        for i in range(self.size):
            for j in range(self.size):
                if talon(self.r_prob):
                    self._field[i][j] = 1
                    self._rabbits.append(self.Rabbit(i, j))
                elif talon(self.w_prob):
                    if talon(self.s_prob):
                        self._field[i][j] = 2
                        self._she_wolves.append(self.She_Wolf(i, j, 10))
                    else:
                        self._field[i][j] = 3
                        self._wolves.append(self.Wolf(i, j, 10))
        self._field_prev = np.zeros((self.size, self.size), dtype='int8')
        return [(len(self._rabbits), 'Кролики'), (len(self._wolves), 'Волки'), (len(self._she_wolves), 'Волчицы')]

    def step(self):
        self._field_prev = self._field.copy()
        self._field = np.zeros((self.size, self.size), dtype='int8')
        for rabbit in self._rabbits:
            i, j = rabbit.coords()
            x = choice((i - 1, i, i + 1))
            y = choice((j - 1, j, j + 1))
            if x > self.size - 1 or x < 0:
                x = i
            if y > self.size - 1 or y < 0:
                y = j
            rabbit.move(x, y)
            self._field[x][y] = 1
            if talon(0.2) and len(self._rabbits) < self.size * self.size // 4:
                self._rabbits.append(self.Rabbit(x, y))

        for wolf in self._she_wolves:
            i, j = wolf.coords()
            if wolf.life <= 0:
                self._she_wolves.remove(wolf)
                continue
            coords = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                      (i, j - 1), (i, j), (i, j + 1),
                      (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
            possible_coords = [coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size)]
            shuffle(possible_coords)
            for x, y in possible_coords:
                if self._field_prev[x][y] == 1:
                    for rabbit in self._rabbits:
                        if rabbit.coords() == (x, y):
                            self._rabbits.remove(rabbit)
                            wolf.life += 10
                    break
            else:
                x, y = choice(possible_coords)
            wolf.move(x, y)
            self._field[x][y] = 2

        for wolf in self._wolves:
            i, j = wolf.coords()
            if wolf.life <= 0:
                self._wolves.remove(wolf)
                continue
            coords = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                      (i, j - 1), (i, j + 1),
                      (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
            possible_coords = [coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size)]
            shuffle(possible_coords)
            for x, y in possible_coords:
                if self._field_prev[x][y] == 1:
                    for rabbit in self._rabbits:
                        if rabbit.coords() == (x, y):
                            self._rabbits.remove(rabbit)
                            wolf.life += 10
                    break
            else:
                for x, y in possible_coords:
                    if self._field_prev[x][y] == 2:
                        for she_wolf in self._she_wolves:
                            if she_wolf.coords() == (x, y) and she_wolf.cooldown <= 0:
                                she_wolf.cooldown = 5
                                if talon(0.5) and len(self._wolves) < self.size:
                                    self._wolves.append(self.Wolf(x, y, 10))
                                elif len(self._she_wolves) < self.size:
                                    self._she_wolves.append(self.She_Wolf(x, y, 10))
                                break
                        break
                else:
                    x, y = choice(possible_coords)
            wolf.move(x, y)
            self._field[x][y] = 3
        return [len(self._rabbits), len(self._wolves), len(self._she_wolves)]


# 2
class Game_of_Life(Abstract_model):
    """Yep, Another one realisation of Conway's 'Game of Life'"""
    MODEL_TEXT = "Игра \"Жизнь\" Конвея. В начале по полю разбросаны случайные живые и мертвые " \
                 "клетки. В мертвых клетка зарождается жизнь, если рядом есть ровно 3 живые " \
                 "клетки. Живые клетки умирают, если рядом меньше 2 (от одиночества) или больше " \
                 "3 (от переначеления) живых соседей."
    COLOR_LIST = ('#808080', '#00ff00')
    PARAMETERS = {'prob': {'value': 0.5, 'min': 0.1, 'max': 0.9, 'spin_type': 'QDoubleSpinBox',
                           'name_rus': 'Процент живых клеток'}}

    def __init__(self, size: int = 10, prob: float = 0.5):
        super().__init__(size)
        self.prob = prob

    def start(self):
        # Random living cells in the world
        self._field = np.zeros((self.size, self.size), dtype='bool')
        for i in range(self.size):
            for j in range(self.size):
                if talon(self.prob):
                    self._field[i][j] = True
        self._field_prev = np.zeros((self.size, self.size), dtype='bool')
        return [(self._field.sum(), 'Живые клетки')]

    def step(self):
        self._field_prev = self._field.copy()
        for i in range(self.size):
            for j in range(self.size):
                neighbours = ((i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                              (i, j - 1), (i, j + 1),
                              (i + 1, j - 1), (i + 1, j), (i + 1, j + 1))
                s = sum((self._field_prev[neighbour[0] % self.size, neighbour[1] % self.size]
                         for neighbour in neighbours))
                self._field[i][j] = ((s == 2) and self._field_prev[i][j]) or (s == 3)
        return [self._field.sum()]


# 3
class Deep_First_Search(Abstract_model):
    """A simulation of DFS"""
    MODEL_TEXT = "Поиск в глубину - один из методов обхода графа. Стратегия поиска в глубину, " \
                 "как и следует из названия, состоит в том, чтобы идти «вглубь» графа, насколько это " \
                 "возможно. Алгоритм поиска описывается рекурсивно: перебираем все исходящие из " \
                 "рассматриваемой вершины рёбра. Если ребро ведёт в вершину, которая не была " \
                 "рассмотрена ранее, то запускаем алгоритм от этой нерассмотренной вершины, а после " \
                 "возвращаемся и продолжаем перебирать рёбра. Возврат происходит в том случае, если в " \
                 "рассматриваемой вершине не осталось рёбер, которые ведут в нерассмотренную вершину."
    COLOR_LIST = ('#644b32', '#c8c8c8', '#ffc800', '#32c832', '#3264ff', '#ffff7d', '#64ffaf')
                # 0 - wall, 1 - empty, 2 - checked, 3 - start, 4 - finish, 5 - next to check, 6 - path
    PARAMETERS = {'p': {'value': 0.33, 'min': 0, 'max': 0.9, 'spin_type': 'QDoubleSpinBox',
                        'name_rus': 'Процент обрушенных стен'}}

    def __init__(self, size: int = 10, p: float = 0.33):
        super().__init__(size)
        self.p = p
        self._parents = {}
        self._finish = (0, 0)
        self._to_check = list()

    def start(self):
        # Field represented as walls but some special cells are empty
        self._field = np.zeros((self.size, self.size), dtype='int8')
        for i in range(1, self.size, 2):
            for j in range(1, self.size, 2):
                self._field[i][j] = 2
        # Then we make a labyrinth. Starting with random cell
        now_i, now_j = choice(range(1, self.size, 2)), choice(range(1, self.size, 2))
        self._field[now_i][now_j] = 1
        to_check = [(now_i, now_j)]
        while to_check:
            # Returning to the last cell
            i, j = to_check.pop()
            coords = ((i - 2, j), (i, j - 2), (i, j + 2), (i + 2, j))
            coords = (coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size))
            coords = [(x, y) for x, y in coords if self._field[x][y] == 2]
            while coords:
                # If it have not checked neighbours
                to_check.append((i, j))
                next_i, next_j = choice(coords)
                # Make path through the walls
                self._field[next_i][next_j] = 1
                self._field[(next_i+i)//2][(next_j+j)//2] = 1
                # Continue with the random neighbour while it have its own neighbours
                i, j = next_i, next_j
                coords = ((i - 2, j), (i, j - 2), (i, j + 2), (i + 2, j))
                coords = (coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size))
                coords = [(x, y) for x, y in coords if self._field[x][y] == 2]

        # Previous labyrinth have only one way to every cell and it's boring
        for _ in range(round(self.size ** 2 * self.p)):
            i, j = choice(range(1, self.size - 1)), choice(range(1, self.size - 1))
            self._field[i][j] = 1
        # Make start and remember its position
        i, j = choice(range(1, self.size, 2)), choice(range(1, self.size, 2))
        self._field[i][j] = 3
        # We have to remember the positions near start in LIFO data structure
        coords = ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j))
        coords = (coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size))
        self._to_check = [(x, y) for x, y in coords if self._field[x][y] == 1]
        for i, j in self._to_check:
            self._field[i][j] = 5
        # Make finish and don't break start
        while self._field[i][j] != 1:
            i, j = choice(range(1, self.size, 2)), choice(range(1, self.size, 2))
        self._field[i][j] = 4
        self._finish = (i, j)
        self._field_prev = np.zeros((self.size, self.size), dtype='int8')
        return []

    def step(self):
        self._field_prev = self._field.copy()
        if self._to_check:
            i, j = self._to_check.pop()
            coords = ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j))
            coords = [coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size)]
            coords = [(x, y) for x, y in coords if self._field[x][y] == 1 or self._field[x][y] == 4]
            shuffle(coords)
            for coord in coords:
                self._parents[coord] = (i, j)
                if self._field[coord[0]][coord[1]] == 4:
                    self._to_check = []
                    break
                else:
                    self._field[coord[0]][coord[1]] = 5
                    self._to_check.append(coord)
            self._field[i][j] = 2
        else:
            prev = self._finish
            while prev in self._parents:
                prev = self._parents[prev]
                self._field[prev[0]][prev[1]] = 6
        return []


# 4
class Breadth_First_Search(Deep_First_Search):
    """A simulation of BFS"""
    MODEL_TEXT = "Поиск в ширину - один из методов обхода графа. Алгоритм поиска в ширину " \
                 "систематически обходит все ближайшие к текущей вершине ребра для «открытия» всех " \
                 "вершин, достижимых из текущей вершины, вычисляя при этом расстояние (минимальное " \
                 "количество рёбер) от текущей вершины до каждой достижимой вршины. Алгоритм " \
                 "работает как для ориентированных, так и для неориентированных графов. Поиск " \
                 "в ширину имеет такое название потому, что в процессе обхода мы идём вширь, т.е. перед " \
                 "тем как приступить к поиску вершин на расстоянии k+1, выполняется обход вершин на " \
                 "расстоянии k. Поиск в ширину является одним из неинформированных алгоритмов поиска."

    def start(self):
        super().start()
        self._to_check = deque(self._to_check)
        return []

    def step(self):
        self._field_prev = self._field.copy()
        if self._to_check:
            i, j = self._to_check.popleft()
            coords = ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j))
            coords = [coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size)]
            coords = [(x, y) for x, y in coords if self._field[x][y] == 1 or self._field[x][y] == 4]
            shuffle(coords)
            for coord in coords:
                self._parents[coord] = (i, j)
                if self._field[coord[0]][coord[1]] == 4:
                    self._to_check = deque([])
                    break
                else:
                    self._field[coord[0]][coord[1]] = 5
                    self._to_check.append(coord)
            self._field[i][j] = 2
        else:
            prev = self._finish
            while prev in self._parents:
                prev = self._parents[prev]
                self._field[prev[0]][prev[1]] = 6
        return []


# 5
class A_Star(Deep_First_Search):
    """A simulation of A*"""
    MODEL_TEXT = "Поиск A* - алгоритм поиска по первому наилучшему совпадению на графе, " \
                 "который находит маршрут с наименьшей стоимостью от одной вершины (начальной) " \
                 "к другой (целевой, конечной). Порядок обхода вершин определяется эвристической функцией " \
                 "«расстояние + стоимость». Эта функция — сумма двух других: функции стоимости достижения " \
                 "рассматриваемой вершины из начальной, и функции эвристической оценки расстояния от рассматриваемой " \
                 "вершины к конечной. В качестве второй функции здесь используется расстояние городских кварталов от " \
                 "рассматриваемой вершины до конечной точки. \nКак правило, этот алгоритм работает чуть быстрее, " \
                 "чем поиск в ширину и находит более короткий путь, чем поиск в глубину. Все зависит от формы графа " \
                 "и выбора эвристических функций."

    def manhattan(self, x, y):
        return abs(x - self._finish[0]) + abs(y - self._finish[1])

    def start(self):
        super().start()
        cop = self._to_check.copy()
        self._to_check = []
        for a in cop:
            heappush(self._to_check, (self.manhattan(*a) + 1, a))
        return []

    def step(self):
        self._field_prev = self._field.copy()
        if self._to_check:
            cost, (i, j) = heappop(self._to_check)
            coords = ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j))
            coords = [coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size)]
            coords = [(x, y) for x, y in coords if self._field[x][y] == 1 or self._field[x][y] == 4]
            shuffle(coords)
            for coord in coords:
                self._parents[coord] = (i, j)
                if self._field[coord[0]][coord[1]] == 4:
                    self._to_check = []
                    break
                else:
                    self._field[coord[0]][coord[1]] = 5
                    new_cost = cost + 1 + self.manhattan(*coord) - self.manhattan(i, j)
                    heappush(self._to_check, (new_cost, coord))
            self._field[i][j] = 2
        else:
            prev = self._finish
            while prev in self._parents:
                prev = self._parents[prev]
                self._field[prev[0]][prev[1]] = 6
        return []


# 6
class Sugar(Abstract_model):
    """Yep, Another one realisation of Conway's 'Game of Life'"""
    MODEL_TEXT = "\"Сахар\"  – это универсальная модель, адаптированная для самых разных тем. В своей простейшей " \
                 "форме \"Сахар\" представляет собой модель простой экономики, в которой агенты перемещаются " \
                 "по двумерной сетке, собирая и накапливая сахар, который представляет собой экономическое " \
                 "благосостояние. Некоторые части сетки производят больше сахара, чем другие, и некоторые агенты " \
                 "находят его лучше, чем другие. Эта версия \"Сахара\" часто используется для изучения и объяснения " \
                 "распределения богатства, в частности тенденции к неравенству. Каждая ячейка поля " \
                 "имеет емкость, которая является максимальным количеством сахара, которое она может содержать. " \
                 "В исходной конфигурации есть две области с высоким содержанием сахара, окруженные " \
                 "концентрическими кольцами с разными емкостями. \n В случайных местах поля расподложены агенты, " \
                 "имеющие три случайно заданных параметра - запас сахара, метаболизм (сколько единиц сахара из " \
                 "запаса поглощается каждый ход) и дальность обзора. Если у агента запас сахара окажется меньше, " \
                 "чем требует метаблизм, то агент погибает. Каждый ход агент рассматривает клетки по горизонтали и " \
                 "вертикали в пределах дальности обзора, затем агент выбирает клетку с максимальным запасом сахара и " \
                 "перемещается туда, собирая сахар в свой запас."
    COLOR_LIST = ('#960000',
                  '#ffffc8', '#ffefcd', '#ffdfd2', '#ffcfd7', '#ffbfdc', '#ffafe1',
                  '#ff9fe6', '#ff8feb', '#ff7ff0', '#ff6ff5', '#ff64ff')
    # 0 - ant
    # from 1 to 11 is amount of sugar minus one
    PARAMETERS = {'capacity': {'value': 70, 'min': 50, 'max': 200, 'spin_type': 'QSpinBox',
                               'name_rus': 'Максимальная вместимость агента'},
                  'metabolism': {'value': 10, 'min': 1, 'max': 24, 'spin_type': 'QSpinBox',
                                 'name_rus': 'Максимально требуемое значение сахара, потребляемое за ход'},
                  'view': {'value': 6, 'min': 1, 'max': 20, 'spin_type': 'QSpinBox',
                           'name_rus': 'Максимальная дальность обзора агента'}}

    class Ant(Special_cell):
        def __init__(self, x, y, sugar, metabolism, view):
            super().__init__(x, y)
            self.sugar = sugar
            self.metabolism = metabolism
            self.view = view

    def __init__(self, size: int = 10, capacity: int = 70, metabolism: int = 10, view: int = 6):
        super().__init__(size)
        self.max_capacity = capacity
        self.max_metabolism = metabolism
        self.max_view = view
        self._ants = []
        self._capacity = self._field.copy()

    def start(self):
        # Random living cells in the world
        self._field = np.ones((self.size, self.size), dtype='int8')
        center_1 = self.size * 3 // 10
        center_2 = self.size * 7 // 10
        rad = round(center_1*1.25)
        for i in range(self.size * 3 // 4):
            for j in range(self.size * 3 // 4):
                self._field[i][j] = max(round((1 - hypot(i - center_1, j - center_1)/rad)*9), 0) + 1
        for i in range(self.size // 4, self.size):
            for j in range(self.size // 4, self.size):
                self._field[i][j] = max(max(round((1 - hypot(i - center_2, j - center_2)/rad)*9), 0) + 1,
                                        self._field[i][j])
        self._capacity = self._field.copy()
        self._field_prev = np.ones((self.size, self.size), dtype='int8')
        for k in range(self.size*self.size // 5):
            x = randint(0, self.size // 2)
            y = randint(0, self.size // 2)
            sugar = randint(10, self.max_capacity)
            view = randint(1, self.max_view)
            metabolism = randint(1, self.max_metabolism)
            ant = self.Ant(x, y, sugar, view, metabolism)
            self._ants.append(ant)
            self._field[x][y] = 0
        return [(len(self._ants), 'Агенты')]

    def step(self):
        # Restore the old ants' positions
        self._field[self._field == 0] = 1
        self._field_prev = self._field.copy()
        # Sugar will regenerate
        self._field[self._field < self._capacity] += 1
        shuffle(self._ants)
        for ant in self._ants:
            if ant.sugar <= 0:
                self._ants.remove(ant)
                continue
            x, y = ant.coords()
            coords = [(x, c) for c in range(y - ant.view, y + ant.view + 1)] +\
                     [(c, y) for c in range(x - ant.view, x + ant.view + 1)]
            possible_coords = [coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size)]
            shuffle(possible_coords)
            # Take the max sugar in sight of view
            next_coords = max(possible_coords, key=lambda c: self._field[c])
            sugar = self._field[next_coords]
            ant.move(*next_coords)
            ant.sugar += sugar - ant.metabolism - 1
            self._field[next_coords] = 0
        return [len(self._ants)]


# 7
class Sand_Pile(Abstract_model):
    """A simulation of sand pile"""
    MODEL_TEXT = "Абстрактная модель песчаной кучи. Модель песчаной кучи – " \
                 "это двумерный клеточный автомат, в котором состояние каждой ячейки " \
                 "представляет склон части кучи песка. В течение каждого временного " \
                 "шага проверяется, не превышает ли каждая ячейка критическое значение. " \
                 "Если это так, она «опрокидывается» и переносит песок в четыре соседние " \
                 "ячейки. По периметру сетки все ячейки находятся на склоне, поэтому избыток " \
                 "перетекает через край. "
    COLOR_LIST = ('#e1e1e1', '#e3dad0', '#e5d3bf', '#e7ccae', '#e9c59d', '#ebbe8c', '#edb77b', '#efb06a', '#f1a959',
                  '#f3a248', '#f59b37', '#f79426', '#f98d15', '#fb8604', '#ff8000')
    PARAMETERS = {'k': {'value': 4, 'min': 4, 'max': 10, 'spin_type': 'QSpinBox',
                        'name_rus': 'Критическое количество песка'},
                  'n': {'value': 4, 'min': 1, 'max': 10, 'spin_type': 'QSpinBox',
                        'name_rus': 'Длина повторяющейся формы'},
                  'p': {'value': 2, 'min': 0.1, 'max': 10, 'spin_type': 'QDoubleSpinBox',
                        'name_rus': 'Степень формы'}}

    def __init__(self, size: int = 10, k: int = 4, n: int = 4, p: float = 2):
        super().__init__(size)
        self.k = k
        self.n = n
        self.p = p

    def start(self):
        self._field = np.random.randint(0, 11, (self.size, self.size), dtype='int8')
        self._field = np.zeros((self.size, self.size), dtype='int8')
        center = self.size // 2
        for i in range(self.size):
            for j in range(self.size):
                self._field[i][j] = 10 - (round((abs(i - center) ** self.p + abs(j - center) ** self.p)
                                                 ** (1 / self.p)) % self.n)
        self._field_prev = np.zeros((self.size, self.size), dtype='int8')
        return [(self._field.sum(), 'Количество песка')]

    def step(self):
        self._field_prev = self._field.copy()
        for i in range(self.size):
            for j in range(self.size):
                # Sand falls from the heights
                if self._field_prev[i][j] >= self.k:
                    self._field[i][j] -= 4
                    # Any possible move. The edging cell are not connecting with the other side
                    coords = ((i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j))
                    possible_coords = (coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size))
                    for coord in possible_coords:
                        self._field[coord] += 1
        return [self._field.sum()]


# 8
class Forest_Fire(Abstract_model):
    """A simulation of forest fire model"""
    MODEL_TEXT = "Модель лесного пожара - пример модели самоорганизованной критичности. Каждая из ячеек квадратного " \
                 "поля может быть пустой, занятой деревом или огнем. Все деревья рядом с горящем деревом загораются " \
                 "на следующий ход. Горящее дерево тухнет на следующий ход. Каждое дерево имеет вероятность " \
                 "самовозгорания, даже если рядом нет горящего дерева. Также есть вероятность роста дерева в " \
                 "пустой ячейке."
    COLOR_LIST = ('#e1e1c8', '#64c832', '#c83232')
    PARAMETERS = {'p': {'value': 0.01, 'min': 0.005, 'max': 0.1, 'spin_type': 'QDoubleSpinBox',
                        'name_rus': 'Вероятность роста дерева'},
                  'f': {'value': 0.001, 'min': 0.001, 'max': 0.01, 'spin_type': 'QDoubleSpinBox',
                        'name_rus': 'Вероятность возгорания'}}
    # 0 - empty, 1 - tree, 2 - fire

    def __init__(self, size: int = 10, p: float = 0.01, f: float = 0.001):
        super().__init__(size)
        self.p = p
        self.f = f

    def start(self):
        self._field = np.random.randint(0, 2, (self.size, self.size), dtype='int8')
        self._field_prev = np.zeros((self.size, self.size), dtype='int8')
        return [(self._field.sum(), 'Деревья'), (0, 'Пожары')]

    def step(self):
        self._field_prev = self._field.copy()
        for i in range(self.size):
            for j in range(self.size):
                # flame will stop at next turn
                if self._field_prev[i][j] == 2:
                    self._field[i][j] = 0
                    coords = ((i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                              (i, j - 1), (i, j + 1),
                              (i + 1, j - 1), (i + 1, j), (i + 1, j + 1))
                    possible_coords = (coord for coord in coords if (min(coord) >= 0 and max(coord) < self.size))
                    for coord in possible_coords:
                        if self._field_prev[coord] == 1:
                            self._field[coord] = 2
                # new tree has chance to grow
                elif self._field_prev[i][j] == 0:
                    if talon(self.p):
                        self._field[i][j] = 1
                # tree has chance to ignite spontaneously
                else:
                    if talon(self.f):
                        self._field[i][j] = 2
        return [len(self._field[self._field == 1]), len(self._field[self._field == 2])]

    def is_ended(self) -> bool:
        return self._field.sum() == 0


# 9
class Segregation(Abstract_model):
    """A simulation of race segregation"""
    MODEL_TEXT = "Модель сегрегации - модель поведения, предложенная Томасом Шеллингом в 1969 году. \nМодель " \
                 "представляет собой сетку, где каждая клетка является домом. Дома могут быть заняты агентами " \
                 "синего или красного цвета, либо быть пустыми. \nВ любой момент времени агент может быть счастлив " \
                 "или несчастлив - зависит от процента соседей того же цвета, что и агент. При моделировании на " \
                 "каждом шаге выбирается случайный агент. Если он несчастлив, то выбирается случайная свободная " \
                 "клетка и агент перемещается туда.\nВ процессе моделирования образуются кластеры одинаковых цветов, " \
                 "что показывает возникновение сегрегации."
    COLOR_LIST = ('#e1e1e1', '#af0000', '#0096ff')
    PARAMETERS = {'p': {'value': 0.3, 'min': 0, 'max': 0.876, 'spin_type': 'QDoubleSpinBox',
                        'name_rus': 'Процент соседей одинакового цвета для счастья агента'},
                  'w': {'value': 0.1, 'min': 0.1, 'max': 0.75, 'spin_type': 'QDoubleSpinBox',
                        'name_rus': 'Доля пустых домов'},
                  'r': {'value': 0.5, 'min': 0.1, 'max': 0.9, 'spin_type': 'QDoubleSpinBox',
                        'name_rus': 'Доля красного цвета среди всех агентов'}}

    # 0 - empty, 1 - red, 2 - blue

    def __init__(self, size: int = 10, p: float = 0.3, w: float = 0.1, r: float = 0.5):
        super().__init__(size)
        self.neighbours_percent = p
        r = (1 - w) * r
        b = 1 - w - r
        self.races_probs = (w, r, b)

    def start(self):
        self._field = np.random.choice((0, 1, 2), (self.size, self.size), p=self.races_probs)
        self._field_prev = np.zeros((self.size, self.size), dtype='int8')
        return [(0, 'Уровень сегрегации')]

    def step(self):
        self._field_prev = self._field.copy()
        total_segregation = 0
        to_move = []
        # Checking for unhappy agents
        for i in range(self.size):
            for j in range(self.size):
                if self._field[i][j]:
                    coords = ((i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                              (i, j - 1), (i, j + 1),
                              (i + 1, j - 1), (i + 1, j), (i + 1, j + 1))
                    # Border are connected
                    neighbours = [self._field_prev[(coord[0] % self.size, coord[1] % self.size)] for coord in coords]
                    same = sum((1 for neighbour in neighbours if neighbour == self._field[i][j]))
                    total = sum((1 for neighbour in neighbours if neighbour != 0))
                    segregation = same / total if total else 1
                    # Counting current segregation level
                    total_segregation += segregation
                    if segregation < self.neighbours_percent:
                        to_move.append((i, j))
        # Random moving unhappy agents
        empty = self._field_prev == 0
        empty_locs = list(zip(*np.nonzero(empty)))
        count_empty = np.sum(empty)
        for coord in to_move:
            k = np.random.randint(count_empty)
            new_coord = empty_locs[k]
            self._field[new_coord], self._field[coord] = self._field[coord], self._field[new_coord]
            empty_locs[k] = coord
        return [total_segregation*100/(self.size ** 2 - count_empty)]


MODELS = {"Инфекция": Ringworm,
          "Волчий остров": Wolf_Island,
          "Игра Жизнь": Game_of_Life,
          "Поиск в глубину": Deep_First_Search,
          "Поиск в ширину": Breadth_First_Search,
          "Поиск A*": A_Star,
          "Сахар": Sugar,
          "Песчаная куча": Sand_Pile,
          "Лесной пожар": Forest_Fire,
          "Сегрегация": Segregation
          }

if __name__ == "__main__":
    def rgb2str(r, g, b):
        r = '0' * (r < 16) + hex(r)[2:]
        g = '0' * (g < 16) + hex(g)[2:]
        b = '0' * (b < 16) + hex(b)[2:]
        return '#' + r + g + b

    def linear_gradient(r1, g1, b1, r2, g2, b2, N):
        dr = (r2 - r1) // (N - 1)
        dg = (g2 - g1) // (N - 1)
        db = (b2 - b1) // (N - 1)
        r, g, b = [r1], [g1], [b1]
        for i in range(N - 2):
            r.append(r[-1] + dr)
            g.append(g[-1] + dg)
            b.append(b[-1] + db)
        r.append(r2)
        g.append(g2)
        b.append(b2)
        return [rgb2str(r[i], g[i], b[i]) for i in range(N)]

    print(rgb2str(225, 225, 200))
    print(rgb2str(100, 200, 50))
    print(rgb2str(200, 50, 50))
    print(linear_gradient(255, 255, 200, 255, 100, 255, 11))
