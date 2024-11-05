from info import ItemList


def check_menu(item_list: ItemList):
    print(f'주문 금액은 총 {item_list.total_price}입니다')
    print(f'주문 품목은 아래와 같습니다')
    for item in item_list.beverages:
        print(f'{item}, ', end='')
    
    print()


def order(item_list: ItemList):
    if len(item_list.beverages) == 0:
        print('주문한 품목이 없습니다.')
        return

    check_menu(item_list)
    print('정말로 주문하시겠습니까?')
    print('1: yes')
    print('2: no')
    choice = int(input('입력: '))

    return True if choice == 1 else False
