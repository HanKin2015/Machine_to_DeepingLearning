#include <iostream>
#include <fstream>
#include <bits/stdc++.h>
#include <cstdio>
using namespace std;

void TestFile()
{
    // C++的读取文件操作
    ifstream inFile;
    inFile.open("c:\\tmp\\test.txt", ios::in);
    if (inFile)  //条件成立，则说明文件打开成功
        inFile.close();
    else
        cout << "test.txt doesn't exist" << endl;
    ofstream oFile;
    oFile.open("test1.txt", ios::out);
    if (!oFile)  //条件成立，则说明文件打开出错
        cout << "error 1" << endl;
    else
        oFile.close();
    oFile.open("tmp\\test2.txt", ios::out | ios::in);
    if (oFile)  //条件成立，则说明文件打开成功
        oFile.close();
    else
        cout << "error 2" << endl;
    fstream ioFile;
    ioFile.open("..\\test3.txt", ios::out | ios::in | ios::trunc);
    if (!ioFile)
        cout << "error 3" << endl;
    else
        ioFile.close();

    // C语言的读取操作
    FILE *fp = NULL;

    fp = fopen("/tmp/test.txt", "w+");
    fprintf(fp, "This is testing for fprintf...\n");
    fputs("This is testing for fputs...\n", fp);
    fclose(fp);

    char buff[255];
    fp = fopen("/tmp/test.txt", "r");
    fscanf(fp, "%s", buff);
    printf("1: %s\n", buff );
    fgets(buff, 255, (FILE*)fp);
    printf("2: %s\n", buff );
    fgets(buff, 255, (FILE*)fp);
    printf("3: %s\n", buff );
    fclose(fp);

    int A[ 3 ][ 10 ];
    int i, j;
    ifstream input( "input.txt", ios::in );
    if( ! input ) {
        cerr << "Open input file error!" << endl;
        exit( -1 );
    }
    ofstream output( "output.txt", ios::out );
    if( ! output ) {
        cerr << "Open output file error!" << endl;
        exit( -1 );
    }
    for( i = 0; i < 3; i++ ) {
        for( j = 0 ; j < 10; j++ ) {
            input >> A[ i ][ j ];
        }
    }
    for( i = 0; i < 3; i++ ) {
        for( j = 0 ; j < 10; j++ ) {
            cout << A[ i ][ j ] << " ";
        }
        cout << endl;
    }
    for( i = 0; i < 3; i++ ) {
        for( j = 0 ; j < 10; j++ ) {
            output << A[ i ][ j ] << " ";
        }
        output << "\r\n";
    }
    input.close();
    output.close();
}

void TestCPJ()
{
    int a[] = {1, 2, 3, 4}, b[] = {3, 4, 5, 6};
    int maxn = 950000;  // 6w需要7.6s，所以60w需要760s，即10分钟，能接受
    long long cnt = 0;
    clockid_t T = clock();
    for (int i = 0; i < maxn; i++) {
        for (int j = 0; j < maxn; j++) {
            cnt++;
        }
    }
    cout << cnt << endl;
    cout << clock() - T << endl;


    // 测试是stl快还是暴力
    set<int> w1, w2;
    for (int i = 0; i < 120000; i++) {
        int n = rand(), m = rand();
        //cout << n << ' ' << m << endl;
        w1.insert(n);
        w2.insert(m);

    }
    clock_t T1 = clock();
    int same = 0;
    for (int i : w1) {
        for (int j : w2) {
            if (i == j) same++;
        }
    }
    cout << "same = " << same << endl;
    cout << clock() - T1 << endl; // 1977

    clockid_t T2 = clock();
    set<int> intersectionS;  //交集
	set_intersection(w1.begin(), w1.end(), w2.begin(), w2.end(), inserter(intersectionS, intersectionS.begin()));


    cout << "same = " << intersectionS.size() << endl;
    cout << clock() - T2 << endl; // 2

    set<int> unionS;  //并集
    set_union(w1.begin(), w1.end(), w2.begin(), w2.end(), inserter(unionS, unionS.begin()));
    cout << "all = " << unionS.size() << endl;
    cout << clock() - T2 << endl;
    return;
}

int main()
{
    //TestCPJ();
    string str = "hj.txt";
    FILE *file = fopen(str.data(), "w");
    fprintf(file, "dasda");
    fclose(file);
    return 0;
}

