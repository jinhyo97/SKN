from show_menu import show_menu
from make_account import make_account
from deposit_money import deposit_money
from withdraw_money import withdraw_money
from show_all_account_information import show_all_account_information

from enum import Enum


class Menu(Enum):
	MAKE = 1
	DEPOSIT = 2
	WITHDRAW = 3
	INQUIRE = 4
	EXIT = 5


def main():
	while True:
		show_menu()
		choice = int(input('선택: '))
		
		if choice == Menu.MAKE.value:
			make_account()
		elif choice == Menu.DEPOSIT.value:
			deposit_money()
		elif choice == Menu.WITHDRAW.value:
			withdraw_money()
		elif choice == Menu.INQUIRE.value:
			show_all_account_information()
		elif choice == Menu.EXIT.value:
			return
		else:
			print("Illegal selection..")


if __name__ == '__main__':
	main()
