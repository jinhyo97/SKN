def menu_input():
    try:
        return int(input("선택: "))
    except:
        return menu_input()
