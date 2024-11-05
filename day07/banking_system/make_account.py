from account import Account
from info import account_list, num_of_accounts


def make_account():
    """Create a new account."""
    global num_of_accounts
    print("[계좌개설]")
    account_id = int(input("계좌ID: "))
    customer_name = input("이  름: ")
    balance = int(input("입금액: "))
    print()

    # Create and add new account
    new_account = Account(account_id, customer_name, balance)
    account_list.append(new_account)
    num_of_accounts += 1
