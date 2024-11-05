from info import account_list


def deposit_money():
    """Deposit money into an account."""
    print("[입    금]")
    account_id = int(input("계좌ID: "))
    money = int(input("입금액: "))

    for account in account_list:
        if account.account_id == account_id:
            account.balance += money
            print("입금완료")
            print()
            return

    print("유효하지 않은 ID 입니다.")
    print()
