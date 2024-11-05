from info import ItemList


def check_menu(item_list: ItemList):
    print(f'주문 금액은 총 {item_list.total_price}입니다')
    print(f'주문 품목은 아래와 같습니다')
    for item in item_list.beverages:
        print(f'{item}, ', end='')
    
    print()