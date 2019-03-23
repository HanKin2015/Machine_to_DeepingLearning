#include<bits/stdc++.h>
using namespace std;
/*
��·�൱���Ǳߣ�·�ھ��Ƕ��㡣
106�ߡ�65�㡢10241��
��65������65*64/2=2080����
���˽��飺ÿ���ļ��ֿ�����ȽϺ�
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

int G[1005][1005];  // 0���ޱߣ���Ϊ0���бߣ�ֵΪ��·���
const int maxRoadId = 1005;
vector<int> edge[maxRoadId];  // ����洢��·�������˵㣬��·�ڱ��
// �ṹ�壺��·������

string folder = "1-map-training-1";
bool ReadFile_road()
{
	string road = folder + "/road.txt";   // 106��
    FILE* F_road = fopen(road.data(), "r");
    int cnt = 0;
    if (!F_road) {
        cout << "The file" << road << "not exist" << endl;
        return 0;
    }
    while(!feof(F_road)) {
        char str[1005];
        //fscanf(F_road, "%s\n", str);  // ������һ���ո��ַ�ʱ������ֹͣ��ȡ
        fgets(str, 1005, (FILE*)F_road);// �������з�
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
        //fscanf(F_car, "%s\n", str);  // ������һ���ո��ַ�ʱ������ֹͣ��ȡ
        fgets(str, 1005, (FILE*)F_car);// �������з�
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

// ·��
bool ReadFile_cross()
{
	string cross = folder + "/cross.txt"; // 65��
    int cnt = 0;
    FILE* F_cross = fopen(cross.data(), "r");
    if (!F_cross) {
        cout << "The file" << cross << "not exist" << endl;
        return 0;
    }
    while(!feof(F_cross)) {
        char str[1005];
        //fscanf(F_cross, "%s\n", str);  // ������һ���ո��ַ�ʱ������ֹͣ��ȡ
        fgets(str, 1005, (FILE*)F_cross);// �������з�
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

