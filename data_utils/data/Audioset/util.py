with open('category_names.txt', 'r') as file:
    data = file.readlines()
    data = [element.replace('\n', '') for element in data]
    print(data)