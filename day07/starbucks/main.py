from info import *
from select import select
from add_menu import add_menu
from remove_menu import remove_menu
from check_menu import check_menu
from order import order


# Main function
def main():
    item_list = ItemList()
    while True:
        select()
        choice = int(input("선택: "))
        print("\n\n")

        if choice == Menu.ADD.value:
            add_menu(item_list)
            print("\n\n")
        elif choice == Menu.REMOVE.value:
            remove_menu(item_list)
            print("\n\n")
        elif choice == Menu.CHECK.value:
            check_menu(item_list)
            print("\n\n")
        elif choice == Menu.ORDER.value:
            if order(item_list):
                print("주문 완료. 프로그램을 종료합니다.")
                break
            else:
                print("주문 보류!")
                print("\n\n")
        elif choice == Menu.END.value:
            print("프로그램을 종료합니다")
            break
        else:
            print("잘못된 입력입니다. 동작을 취소합니다.")
            break

if __name__ == "__main__":
    main()