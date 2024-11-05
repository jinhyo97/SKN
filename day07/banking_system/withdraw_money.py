from info import account_list


def withdraw_money():
    """Withdraw money from an account."""
    print("[출    금]")
    account_id = int(input("계좌ID: "))
    money = int(input("출금액: "))

    for account in account_list:
        if account.account_id == account_id:
            if account.balance < money:
                print("잔액부족")
                print()
                return

            account.balance -= money
            print("출금완료")
            print()
            return

    print("유효하지 않은 ID 입니다.")
    print()
