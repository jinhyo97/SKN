from info import account_list


def show_all_account_information():
    """Display all account information."""
    for account in account_list:
        print(f"계좌ID: {account.account_id}")
        print(f"이  름: {account.customer_name}")
        print(f"잔  액: {account.balance}")
        print()
