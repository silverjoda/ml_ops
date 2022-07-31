#include <bits/stdc++.h>
#include <algorithm>
#include <numeric>

using namespace std;

string ltrim(const string &);
string rtrim(const string &);
vector<string> split(const string &);

/*
 * Complete the 'matrixRotation' function below.
 *
 * The function accepts following parameters:
 *  1. 2D_INTEGER_ARRAY matrix
 *  2. INTEGER r
 */

 void rotate(vector<vector<int>>* matrix_A, vector<vector<int>>* matrix_B){
    int M = (*matrix_A).size();
    int N = (*matrix_A)[0].size();

    int m_midpoint = M / 2;
    int n_midpoint = N / 2;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            if (m >= m_midpoint){
                if (n >= n_midpoint){
                    // Right, up
                    if (n < (N - 1 - (M - 1 - m))){
                        // Move right
                        (*matrix_B)[m][n+1] = (*matrix_A)[m][n];
                    }else {
                        (*matrix_B)[m - 1][n] = (*matrix_A)[m][n];
                    }
                }else{
                    // Right, down
                    if (m < (M - 1 - n)){
                        // Move down
                        (*matrix_B)[m + 1][n] = (*matrix_A)[m][n];
                    }else{
                        // Move right
                        (*matrix_B)[m][n + 1] = (*matrix_A)[m][n];
                    }

                }
            }else{
                if (n >= n_midpoint){
                    // left, up
                    if (m > (N - 1 - n)){
                        // Move up
                        (*matrix_B)[m - 1][n] = (*matrix_A)[m][n];
                    }else {
                        // Move left
                        (*matrix_B)[m][n - 1] = (*matrix_A)[m][n];
                    }
                }else{
                    // Left, down
                    if (n > m){
                        // Move left
                        (*matrix_B)[m][n - 1] = (*matrix_A)[m][n];
                    }else{
                        // Move down
                        (*matrix_B)[m + 1][n] = (*matrix_A)[m][n];
                    }

                }
            }
        }
    }
 }


void matrixRotation(vector<vector<int>> matrix, int r) {
    int M = (matrix).size();
    int N = (matrix)[0].size();

    int m_midpoint = M / 2;
    int n_midpoint = N / 2;

    // filter redundant rotations
    int n_layers = min(N - n_midpoint, M - m_midpoint);
    vector<int> multiples;
    for(int i = 1; i <= n_layers; i++){
        multiples.push_back(2 * (M - i) + 2 * (N - i));
    }

    // Least common multiple of all layers
    int lcm = 1;
    for (int i=0; i < multiples.size(); i++){
        lcm = lcm * multiples[i] / __gcd(lcm, multiples[i]);
    }

    r = r % lcm;

    vector<vector<int>> matrix_A(matrix);
    vector<vector<int>> matrix_B(matrix);

    for (int i = 0; i < r; i++) {
        if (i % 2 == 0){
            rotate(&matrix_A, &matrix_B);
        }else{
            rotate(&matrix_B, &matrix_A);
        }
    }

    vector<vector<int>> printmat;
    if(r % 2 == 0){
        printmat = matrix_A;
    }else{
        printmat = matrix_B;
    }

    // Print matrix to output
    for (int i = 0; i < printmat.size(); i++) {
        for (int j = 0; j < printmat[0].size(); j++) {
            printf( "%i ", printmat[i][j]);
        }
        printf("\n");
    }

}

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    string first_multiple_input_temp;
    getline(cin, first_multiple_input_temp);

    vector<string> first_multiple_input = split(rtrim(first_multiple_input_temp));

    int m = stoi(first_multiple_input[0]);

    int n = stoi(first_multiple_input[1]);

    int r = stoi(first_multiple_input[2]);

    vector<vector<int>> matrix(m);

    for (int i = 0; i < m; i++) {
        matrix[i].resize(n);

        string matrix_row_temp_temp;
        getline(cin, matrix_row_temp_temp);

        vector<string> matrix_row_temp = split(rtrim(matrix_row_temp_temp));

        for (int j = 0; j < n; j++) {
            int matrix_row_item = stoi(matrix_row_temp[j]);

            matrix[i][j] = matrix_row_item;
        }
    }

    matrixRotation(matrix, r);

    return 0;
}

string ltrim(const string &str) {
    string s(str);

    s.erase(
        s.begin(),
        find_if(s.begin(), s.end(), not1(ptr_fun<int, int>(isspace)))
    );

    return s;
}

string rtrim(const string &str) {
    string s(str);

    s.erase(
        find_if(s.rbegin(), s.rend(), not1(ptr_fun<int, int>(isspace))).base(),
        s.end()
    );

    return s;
}

vector<string> split(const string &str) {
    vector<string> tokens;

    string::size_type start = 0;
    string::size_type end = 0;

    while ((end = str.find(" ", start)) != string::npos) {
        tokens.push_back(str.substr(start, end - start));

        start = end + 1;
    }

    tokens.push_back(str.substr(start));

    return tokens;
}
