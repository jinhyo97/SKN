correct_number = 11
count = 0
while True:
    while True:
        user_input_value = input('숫자를 입력하세요: ')

        if user_input_value.isdecimal():
            break
        else:
            print('숫자로만 입력해주세요.')
    
    user_input_value = int(user_input_value)

    count += 1
    if user_input_value > correct_number:
        print('Down')
    elif user_input_value == correct_number:
        print('정답')
        print(f'정답까지 소요된 횟수: {count}')
        break
    else:
        print('Up')