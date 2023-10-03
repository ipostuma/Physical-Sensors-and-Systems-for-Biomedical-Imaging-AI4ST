# Lecture 0

The lecture is focussed on basic knowledge about the GNU/linux system and BASH shell interface.

## How to use grep, awk, and sed

Hands-on example that demonstrates how to use grep, awk, and sed to perform text processing tasks. 
In this example, we'll work with a sample text file containing a list of employees and their salaries.

Assume you have a text file named employees.txt with the following content:

```
John Doe, $50000
Jane Smith, $60000
Bob Johnson, $55000
Alice Brown, $58000
```

### Task 1: Using grep to Search for Patterns
Objective: Use grep to find employees with salaries greater than $55,000.

<details>
  <summary>Answer</summary>

```
grep '\$[5-9][0-9]\{4\}' employees.txt
```

Explanation:

* \ is used to escape the $ symbol because it has a special meaning in regular expressions.
* [5-9] matches any digit from 5 to 9.
* [0-9]\{4\} matches exactly four digits (the salary part).
  
</details>

Correct result:

```
Jane Smith, $60000
Alice Brown, $58000
```

### Task 2: Using awk to Extract and Manipulate Data

Objective: Use awk to extract the employee names and salaries and calculate their bonuses (10% of the salary).

<details>
  <summary>Answer</summary>

```
awk -F', ' '{printf "Name: %s, Salary: $%d, Bonus: $%d\n", $1, $2, $2*0.1}' employees.txt
```

Explanation:

* -F', ' sets the field separator as a comma followed by a space.
* '{printf "Name: %s, Salary: $%d, Bonus: $%d\n", $1, $2, $2*0.1}' formats and prints the output.

</details>

Correct result:

```
Name: John Doe, Salary: $50000, Bonus: $5000
Name: Jane Smith, Salary: $60000, Bonus: $6000
Name: Bob Johnson, Salary: $55000, Bonus: $5500
Name: Alice Brown, Salary: $58000, Bonus: $5800
```

### Task 3: Using sed to Replace Text

Objective: Use sed to replace the dollar sign $ with the word "USD" in the salary amounts.

<details>
  <summary>Answer</summary>

```
sed 's/\$/USD/g' employees.txt
```

Explanation:

* s/\$/USD/g is a sed substitution command that replaces all occurrences of $ with "USD."

</details>

Correct result:

```
John Doe, USD50000
Jane Smith, USD60000
Bob Johnson, USD55000
Alice Brown, USD58000
```

In this hands-on example, we've demonstrated how to use grep to search for patterns, awk to extract and manipulate data, and sed to replace text within a sample text file. These tools are incredibly versatile and can be applied to various text processing tasks in a Linux environment.

