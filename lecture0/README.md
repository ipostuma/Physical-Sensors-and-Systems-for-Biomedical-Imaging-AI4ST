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

###Task 2: Using awk to Extract and Manipulate Data

Objective: Use awk to extract the employee names and salaries and calculate their bonuses (10% of the salary).
