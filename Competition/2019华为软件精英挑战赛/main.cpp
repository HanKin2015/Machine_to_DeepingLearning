#include<bits/stdc++.h>
using namespace std;
/*
道路相当于是边，路口就是顶点。
106边、65点、10241车
即65个点有65*64/2=2080条边
个人建议：每个文件分开处理比较好
*/

void Split(const string str, vector<int>& v)
{
	int len = str.size();
	int value = 0, isNeg = 1;
    for (int i = 0; i < len; i++) {
        if (str[i] == '-') isNeg = -1;
        else if (str[i] >= '0' && str[i] <= '9') value = value*10 + str[i] - '0';
        else {
            if (value != 0) {
                value *= isNeg;
                v.push_back(value);
                value = 0, isNeg = 1;
            }
        }
    }
    return;
}

int G[1005][1005];  // 0则无边，不为0即有边，值为道路编号
const int maxRoadId = 1005;
vector<int> edge[maxRoadId];  // 里面存储道路的两个端点，即路口编号
// 结构体：道路、汽车

string folder = "1-map-training-1";
bool ReadFile_road()
{
	string road = folder + "/road.txt";   // 106边
    FILE* F_road = fopen(road.data(), "r");
    int cnt = 0;
    if (!F_road) {
        cout << "The file" << road << "not exist" << endl;
        return 0;
    }
    while(!feof(F_road)) {
        char str[1005];
        //fscanf(F_road, "%s\n", str);  // 遇到第一个空格字符时，它会停止读取
        fgets(str, 1005, (FILE*)F_road);// 包括换行符
        if (str[0] == '#') continue;
        vector<int> v;
        Split(str, v);
        cout << str << endl;
        for (int i: v) {
            cout << i << ' ';
        }
        cout << endl;
        cnt++;
    }
    cout << cnt << endl;
    fclose(F_road);
    return true;
}

bool ReadFile_car()
{
	string car = folder + "/car.txt";     // 10241
    int cnt = 0;
    FILE* F_car = fopen(car.data(), "r");
    if (!F_car) {
        cout << "The file" << car << "not exist" << endl;
        return 0;
    }
    while(!feof(F_car)) {
        char str[1005];
        //fscanf(F_car, "%s\n", str);  // 遇到第一个空格字符时，它会停止读取
        fgets(str, 1005, (FILE*)F_car);// 包括换行符
        //cout << str << endl;
        if (str[0] == '#') continue;
        vector<int> v;
        Split(str, v);
        cout << str << endl;
        for (int i: v) {
            cout << i << ' ';
        }
        cout << endl;
        cnt++;
    }
    cout << cnt << endl;
    fclose(F_car);
    return true;
}

// 路口
bool ReadFile_cross()
{
	string cross = folder + "/cross.txt"; // 65点
    int cnt = 0;
    FILE* F_cross = fopen(cross.data(), "r");
    if (!F_cross) {
        cout << "The file" << cross << "not exist" << endl;
        return 0;
    }
    while(!feof(F_cross)) {
        char str[1005];
        //fscanf(F_cross, "%s\n", str);  // 遇到第一个空格字符时，它会停止读取
        fgets(str, 1005, (FILE*)F_cross);// 包括换行符
        if (str[0] == '#') continue;
        vector<int> v;
        Split(str, v);
        cout << str << endl;
        for (int i: v) {
            cout << i << ' ';
        }
        cout << endl;
        cnt++;
    }
    cout << cnt << endl;
    fclose(F_cross);
    return true;
}

void Init()
{
    ReadFile_road();
    //ReadFile_car();
    //ReadFile_cross();
    return;
}

int main()
{
    Init();
	return 0;
}

