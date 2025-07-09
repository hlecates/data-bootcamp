from typing import List, Dict, Tuple, Optional, Union


def demonstrate_data_types():
    # Integer
    age = 25
    print(f"Age: {age}, Type: {type(age)}")
    
    # Float
    height = 5.9
    print(f"Height: {height}, Type: {type(height)}")
    
    # String
    name = "Alice"
    print(f"Name: {name}, Type: {type(name)}")
    
    # Boolean
    is_student = True
    print(f"Is student: {is_student}, Type: {type(is_student)}")
    
    # None
    empty_value = None
    print(f"Empty value: {empty_value}, Type: {type(empty_value)}")
    
    # Type conversion examples
    number_string = "123"
    number_int = int(number_string)
    print(f"String to int: {number_string} -> {number_int}")
    
    pi_float = 3.14159
    pi_int = int(pi_float)
    print(f"Float to int: {pi_float} -> {pi_int}")


def type_checking_examples():
    values = [42, "hello", 3.14, True, None]
    
    for value in values:
        print(f"Value: {value}, Type: {type(value)}")
        
        # Type checking
        if isinstance(value, int):
            print(f"  -> {value} is an integer")
        elif isinstance(value, str):
            print(f"  -> {value} is a string")
        elif isinstance(value, float):
            print(f"  -> {value} is a float")
        elif isinstance(value, bool):
            print(f"  -> {value} is a boolean")
        elif value is None:
            print(f"  -> {value} is None")


def arithmetic_operations():
    
    a, b = 10, 3
    
    operations = {
        "Addition": a + b,
        "Subtraction": a - b,
        "Multiplication": a * b,
        "Division": a / b,
        "Floor Division": a // b,
        "Modulo": a % b,
        "Exponentiation": a ** b
    }
    
    for operation, result in operations.items():
        print(f"{operation}: {a} {operation.lower()} {b} = {result}")


def comparison_operations():
    
    x, y = 5, 10
    
    comparisons = [
        (x == y, "Equal to"),
        (x != y, "Not equal to"),
        (x < y, "Less than"),
        (x > y, "Greater than"),
        (x <= y, "Less than or equal"),
        (x >= y, "Greater than or equal")
    ]
    
    for result, description in comparisons:
        print(f"{x} {description} {y}: {result}")


def logical_operations():
    
    
    is_adult = True
    has_license = False
    has_car = True
    
    print(f"Is adult: {is_adult}")
    print(f"Has license: {has_license}")
    print(f"Has car: {has_car}")
    
    # Logical AND
    can_drive_and = is_adult and has_license
    print(f"Can drive (AND): {can_drive_and}")
    
    # Logical OR
    can_drive_or = is_adult or has_license
    print(f"Can drive (OR): {can_drive_or}")
    
    # Logical NOT
    not_adult = not is_adult
    print(f"Not adult: {not_adult}")
    
    # Complex logical expression
    can_get_loan = is_adult and (has_license or has_car)
    print(f"Can get loan: {can_get_loan}")


def list_examples():
    
    # Creating lists
    numbers = [1, 2, 3, 4, 5]
    mixed = [1, "hello", 3.14, True]
    empty_list = []
    
    print(f"Numbers: {numbers}")
    print(f"Mixed: {mixed}")
    print(f"Empty: {empty_list}")
    
    # Accessing elements
    print(f"First element: {numbers[0]}")
    print(f"Last element: {numbers[-1]}")
    print(f"Slice [1:4]: {numbers[1:4]}")
    print(f"Slice [::2]: {numbers[::2]}")  # Every second element
    
    # List operations
    numbers.append(6)
    print(f"After append: {numbers}")
    
    numbers.insert(0, 0)
    print(f"After insert: {numbers}")
    
    removed = numbers.pop()
    print(f"Removed: {removed}, List: {numbers}")
    
    numbers.remove(2)
    print(f"After remove(2): {numbers}")
    
    # List methods
    print(f"Length: {len(numbers)}")
    print(f"Sum: {sum(numbers)}")
    print(f"Max: {max(numbers)}")
    print(f"Min: {min(numbers)}")
    
    # List comprehension
    squares = [x**2 for x in numbers]
    print(f"Squares: {squares}")
    
    even_squares = [x**2 for x in numbers if x % 2 == 0]
    print(f"Even squares: {even_squares}")


def tuple_examples():
    
    # Creating tuples
    coordinates = (10, 20)
    person = ("Alice", 25, "Engineer")
    single_item = (42,)  # Note the comma!
    
    print(f"Coordinates: {coordinates}")
    print(f"Person: {person}")
    print(f"Single item: {single_item}")
    
    # Accessing elements
    print(f"X coordinate: {coordinates[0]}")
    print(f"Person's name: {person[0]}")
    print(f"Person's age: {person[1]}")
    
    # Tuple unpacking
    x, y = coordinates
    name, age, job = person
    print(f"X: {x}, Y: {y}")
    print(f"Name: {name}, Age: {age}, Job: {job}")
    
    # Tuple methods
    repeated_tuple = (1, 2, 2, 3, 2, 4)
    print(f"Count of 2: {repeated_tuple.count(2)}")
    print(f"Index of 3: {repeated_tuple.index(3)}")


def dictionary_examples():
    
    # Creating dictionaries
    person = {
        "name": "Alice",
        "age": 25,
        "city": "New York",
        "skills": ["Python", "Data Science", "Machine Learning"]
    }
    
    print(f"Person: {person}")
    
    # Accessing values
    print(f"Name: {person['name']}")
    print(f"Age: {person['age']}")
    print(f"Skills: {person['skills']}")
    
    # Safe access with .get()
    phone = person.get("phone", "Not provided")
    print(f"Phone: {phone}")
    
    # Adding/updating values
    person["email"] = "alice@example.com"
    person["age"] = 26
    print(f"Updated person: {person}")
    
    # Dictionary methods
    print(f"Keys: {list(person.keys())}")
    print(f"Values: {list(person.values())}")
    print(f"Items: {list(person.items())}")
    
    # Dictionary comprehension
    squares_dict = {x: x**2 for x in range(5)}
    print(f"Squares dictionary: {squares_dict}")
    
    # Nested dictionaries
    employees = {
        "alice": {"age": 25, "salary": 50000, "department": "Engineering"},
        "bob": {"age": 30, "salary": 60000, "department": "Marketing"},
        "charlie": {"age": 35, "salary": 70000, "department": "Engineering"}
    }
    
    print(f"Employees: {employees}")
    
    # Accessing nested data
    alice_salary = employees["alice"]["salary"]
    print(f"Alice's salary: {alice_salary}")


def set_examples():

    # Creating sets
    fruits = {"apple", "banana", "orange", "apple"}  # Duplicate removed
    numbers = {1, 2, 3, 4, 5}
    
    print(f"Fruits: {fruits}")
    print(f"Numbers: {numbers}")
    
    # Set operations
    set1 = {1, 2, 3, 4}
    set2 = {3, 4, 5, 6}
    
    print(f"Set 1: {set1}")
    print(f"Set 2: {set2}")
    print(f"Union: {set1 | set2}")
    print(f"Intersection: {set1 & set2}")
    print(f"Difference: {set1 - set2}")
    print(f"Symmetric difference: {set1 ^ set2}")
    
    # Adding and removing
    fruits.add("grape")
    fruits.remove("banana")
    print(f"Updated fruits: {fruits}")
    
    # Set comprehension
    unique_lengths = {len(name) for name in ["Alice", "Bob", "Charlie", "David"]}
    print(f"Unique name lengths: {unique_lengths}")


def if_statement_examples():
    
    # Simple if statement
    age = 18
    
    if age >= 18:
        print("You are an adult")
    elif age >= 13:
        print("You are a teenager")
    else:
        print("You are a child")
    
    # Complex conditions
    income = 50000
    credit_score = 750
    has_collateral = True
    
    if income > 40000 and credit_score > 700:
        print("Loan approved!")
    elif has_collateral and credit_score > 650:
        print("Loan approved with collateral")
    else:
        print("Loan denied")
    
    # Nested if statements
    temperature = 75
    humidity = 60
    
    if temperature > 80:
        if humidity > 70:
            print("Hot and humid - stay inside!")
        else:
            print("Hot but dry - good for outdoor activities")
    elif temperature > 60:
        print("Pleasant weather")
    else:
        print("Cold weather - wear a jacket")


def for_loop_examples():
    
    # Looping through a list
    fruits = ["apple", "banana", "orange"]
    print("Fruits:")
    for fruit in fruits:
        print(f"  - I like {fruit}")
    
    # Looping with range
    print("\nCounting from 0 to 4:")
    for i in range(5):
        print(i, end=" ")
    
    print("\n\nCounting from 1 to 10:")
    for i in range(1, 11):
        print(i, end=" ")
    
    print("\n\nEven numbers from 0 to 10:")
    for i in range(0, 11, 2):
        print(i, end=" ")
    
    # Looping through dictionary
    person = {"name": "Alice", "age": 25, "city": "NYC"}
    print("\n\nPerson details:")
    for key, value in person.items():
        print(f"  {key}: {value}")
    
    # Enumerate example
    print("\nEnumerated list:")
    for index, fruit in enumerate(fruits, 1):
        print(f"  {index}. {fruit}")
    
    # List comprehension with for loop
    numbers = [1, 2, 3, 4, 5]
    doubled = [num * 2 for num in numbers]
    print(f"\nDoubled numbers: {doubled}")


def while_loop_examples():
    
    # Simple while loop
    count = 0
    print("Counting with while loop:")
    while count < 5:
        print(f"Count: {count}")
        count += 1
    
    # While loop with break
    print("\nGuessing game:")
    secret_number = 7
    attempts = 0
    
    while True:
        attempts += 1
        guess = attempts  # Simulating user input
        
        if guess == secret_number:
            print(f"Correct! Found in {attempts} attempts")
            break
        elif guess > secret_number:
            print("Too high!")
        else:
            print("Too low!")
        
        if attempts >= 10:
            print("Game over!")
            break
    
    # While loop with continue
    print("\nPrinting odd numbers:")
    num = 0
    while num < 10:
        num += 1
        if num % 2 == 0:
            continue  # Skip even numbers
        print(num, end=" ")


def file_handling_examples():
    
    # Writing to a file
    sample_data = [
        "Hello,",
        "World!"
    ]
    
    with open('sample.txt', 'w') as file:
        for line in sample_data:
            file.write(line + '\n')
    
    print("File 'sample.txt' written successfully!")
    
    # Reading from a file
    print("\nReading file content:")
    with open('sample.txt', 'r') as file:
        content = file.read()
        print(content)
    


def list_comprehension_examples():
    
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Basic list comprehension
    squares = [x**2 for x in numbers]
    print(f"Squares: {squares}")
    
    # List comprehension with condition
    even_squares = [x**2 for x in numbers if x % 2 == 0]
    print(f"Even squares: {even_squares}")
    
    # Nested list comprehension
    matrix = [[i + j for j in range(3)] for i in range(3)]
    print(f"Matrix: {matrix}")
    
    # Dictionary comprehension
    squares_dict = {x: x**2 for x in range(5)}
    print(f"Squares dictionary: {squares_dict}")
    
    # Set comprehension
    unique_lengths = {len(name) for name in ["Alice", "Bob", "Charlie", "David"]}
    print(f"Unique name lengths: {unique_lengths}")


def main():
    
    # Run all examples
    print("\n1. DATA TYPES EXAMPLES")
    demonstrate_data_types()
    type_checking_examples()
    print("\n")
    
    print("\n2. BASIC OPERATIONS EXAMPLES")
    arithmetic_operations()
    comparison_operations()
    logical_operations()
    print("\n")
    
    print("\n3. DATA STRUCTURES EXAMPLES")
    list_examples()
    tuple_examples()
    dictionary_examples()
    set_examples()
    print("\n")
    
    print("\n4. CONDITIONALS & LOOPS EXAMPLES")
    if_statement_examples()
    for_loop_examples()
    while_loop_examples()
    print("\n")
    
    print("\n5. LIST COMPREHENSIONS EXAMPLES")
    list_comprehension_examples()
    print("\n")
    


if __name__ == "__main__":
    main()
