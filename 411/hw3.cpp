/// hw3.cpp
//
//  Author: Daniel Choo
//  Date:   09/16/19

#include <iostream>
#include <vector>
using namespace std;

int main() {

	// Declaring arrays, variables.
	vector<int> A = {1, 2, 3, 4};
	vector<int> B = {9, 8, 7, 6};
	vector<int> product;

	int x = A.size();
	int counter = 0;
	int C = 0;

	// Beginning of AB product.
	for (int i = 0; i < x; ++i) C+=(A[i]*B[i]);
	cout << "C = " << C << "\n\n";

	// Beginning of BA product.
	printf("%s", "D = [");

	// Creating the matrix product.
	for (int i = 0; i < x; ++i) {
		for (int j = 0; j < x; ++j) {

			// Formatting.
			if (j == 0 && i > 0) printf("%s", "     ");

			// Creating matrix.
			product.push_back(B[i] * A[j]);
			printf("%i", product[counter]);

			// Formatting.
			if ((j%3 != 0) || j == 0) printf("%s", ", ");
			else if (j == 3) printf("%s", "");
			++counter;
		}
 
		if (i < 3) printf("%s", "\n");
		else 	printf("%s", "]");
	}

	// Stripping contents from vector.
	A.clear();
	B.clear();
	product.clear();
		
	return 0;
}
