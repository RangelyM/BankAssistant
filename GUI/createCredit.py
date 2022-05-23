from typing import List, Tuple

import numpy as np
import pandas as pd

# residual = income - minLive

class CreateCredit:
    def __init__(self, credit_sum: int, age: int, residual: int,
                 max_age_threshold=80, percent_range: Tuple[float, float] = (10., 15.)) -> None:
        self.credit_sum = credit_sum
        self.age = age
        self.residual = residual
        self.max_age_threshold = max_age_threshold
        self.percent_range = percent_range

    def calc_per_month(self, month_percent: int, payments_amount: int, expected_sum) -> float:
        return ((month_percent * (month_percent + 1) ** payments_amount) /
                ((month_percent + 1) ** payments_amount - 1)) * self.credit_sum

    def calc_params_of_credit(self, expected_sum: int) :
        # percent variants in ascending order: the lower percent is, the better
        credit_percents = np.arange(*self.percent_range, 0.1)

        # initial iteration: the minimum value of the month's quantity can be only 2
        payments_amount = 2

        # main loop
        while payments_amount / 12 + self.age < self.max_age_threshold:
            for percent in credit_percents:

                # calculate payment_per_month
                payment_per_month = self.calc_per_month(percent / 12, payments_amount, expected_sum)

                # check the conditions
                if (payment_per_month < self.residual) and (payments_amount / 12 + self.age < self.max_age_threshold):
                    return [expected_sum, payment_per_month, percent, payments_amount]

            # years are varied
            payments_amount += 1

        print('Невозможно выдать кредит на запрошенную сумму')
        return [0] * 4

    def calc_results(self) -> pd.DataFrame:
        delta_sum = self.credit_sum * 0.05  # vary the sum of credit with step 5% of credit sum
        sums_of_credit = [self.credit_sum - i * delta_sum for i in range(10)]

        Bank_helper_result = pd.DataFrame({'Sum of credit': [''],
                                           'Payment per month': [''],
                                           'Bank percent': [''],
                                           'Month credit': [''],
                                           'Overpayment': ['']})

        for dif_sum in sums_of_credit:
            current_result = self.calc_params_of_credit(dif_sum)
            if current_result[0]:
                Bank_helper_result.loc[len(Bank_helper_result.index)] = [*current_result,
                                                                         abs(current_result[0] - current_result[1] *
                                                                             current_result[3])]
        return Bank_helper_result










