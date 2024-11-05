from info import *


def show_menu():
    print('======= Add Menu =======')
    print('1. 아메리카노: 4500')
    print('2. 카페라떼: 5000')
    print('3. 콜드브루: 4900')
    print('4. 에스프레소: 4000')
    print('5. 아이스티: 5900')
    print('6. 말차라떼: 6100')


def add_menu(item_list: ItemList):
    if len(item_list.beverages) > MAX_LEN:
        print('더 이상 주문할 수 없습니다')
        return

    show_menu()
    choice = int(input('선택: '))

    coffee_menu_map = {
        CoffeeMenu.AMERICANO.value: (CoffeeMenu.AMERICANO.name, CoffeePrice.AMERICANO.value),
        CoffeeMenu.CAFE_LATTE.value: (CoffeeMenu.CAFE_LATTE.name, CoffeePrice.CAFE_LATTE.value),
        CoffeeMenu.COLD_BREW.value: (CoffeeMenu.COLD_BREW.name, CoffeePrice.COLD_BREW.value),
        CoffeeMenu.ESPRESSO.value: (CoffeeMenu.ESPRESSO.name, CoffeePrice.ESPRESSO.value),
        CoffeeMenu.ICE_TEA.value: (CoffeeMenu.ICE_TEA.name, CoffeePrice.ICE_TEA.value),
        CoffeeMenu.GREEN_TEA.value: (CoffeeMenu.GREEN_TEA.name, CoffeePrice.GREEN_TEA.value),
    }

    if choice in coffee_menu_map:
        coffee_name, coffee_price = coffee_menu_map.get(choice)
        print(f'{coffee_name} 추가')
        item_list.beverages.append(coffee_name)
        item_list.total_price += coffee_price
    else:
        print('잘못된 입력입니다. 동작을 취소합니다.')
