# Housing loan calculator

This program helps you optimize your fixed deposits while paying off your mortgage.

An example of use case is the following. Suppose, at the moment when you take the loan to purchase a property, or buy anything, there are some fixed deposits which you can use to pay the mortgage upon maturation. You can save the matured deposits back into the bank, as long as there is enough cash to make the monthly mortgage payment. You want to find out whether you can eventually pay off the mortgage with your savings, or, if not, for how long you can maintain the mortgage payment.

This problem can be cast as a type of mathematical problem called linear programming, which is well-understood and admits efficient algorithms. Under the hood, this program simply calls scipy's linprog method to find an optimal solution. Crucially, the program assumes fixed exchange rates and interest rates over the entire tenure of the loan, which can be a major source of inaccuracy.

To start the program, make sure all packages in the requirement.txt are installed, open the terminal and run `voila main.ipynb'.
