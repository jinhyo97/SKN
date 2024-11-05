from enum import Enum


MAX_LEN = 10

class Menu(Enum):
    ADD = 1
    REMOVE = 2
    CHECK = 3
    ORDER = 4
    END = 5


class CoffeeMenu(Enum):
    AMERICANO = 1
    CAFE_LATTE = 2
    COLD_BREW = 3
    ESPRESSO = 4
    ICE_TEA = 5
    GREEN_TEA = 6


class CoffeePrice(Enum):
    AMERICANO = 4500
    CAFE_LATTE = 5000
    COLD_BREW = 4900
    ESPRESSO = 4000
    ICE_TEA = 5900
    GREEN_TEA = 6100   


class ItemList:
    def __init__(self):
        self.beverages = []
        self.total_price = 0