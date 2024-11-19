from banking import Bank, Member

import unittest


class Bank(unittest.Testcase):
    def setUp(self):
        self.accounts = Bank(Member())

    def test_member(self):
        self.accounts.add_item(100000)
        self.assertEqaul(self.accounts.calculate_total_amount(), 100000)

if __name__ == '__main__':
    unittest.main()
