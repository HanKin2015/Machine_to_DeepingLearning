#include<bits/stdc++.h>
using namespace std;
/*
道路相当于是边，路口就是顶点。
106边、64点、10241车
即65个点有65*64/2=2080条边
个人建议：每个文件分开处理比较好

中文输出容易出现乱码问题，建议英文
注意：道路id并不是按顺序编号的
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
const int maxRoadId = 100005, maxCarId = 100005;
vector<int> edge[maxRoadId];  // 里面存储道路的两个端点，即路口编号
set<int> V, E;

// 结构体：道路、汽车
struct S_Road
{
    int id, length, maxSpeed, laneNumber, startId, endId, isDoubleSide;
}road[maxRoadId];
struct S_car
{
    int id, maxSpeed, startLoacation, endLocation, startTime;
}car[maxCarId];

string folder = "1-map-training-1";   // 数据文件夹
bool ReadFile_road()
{
    string roadPath = folder + "/road.txt";   // 106边
    FILE* F_road = fopen(roadPath.data(), "r");
    int cnt = 0;
    if (!F_road) {
        cout << "The file" << roadPath << "not exist" << endl;
        return 0;
    }
    memset(G, 0, sizeof(G));
    while(!feof(F_road)) {
        char str[1005];
        //fscanf(F_road, "%s\n", str);  // 遇到第一个空格字符时，它会停止读取
        fgets(str, 1005, (FILE*)F_road);// 包括换行符
        if (str[0] == '#') continue;
        vector<int> v;  // #(道路id，道路长度，最高限速，车道数目，起始点id，终点id，是否双向)
        Split(str, v);
        int id = v[0];
        E.insert(id);
        road[id].id = v[0];
        road[id].length = v[1];
        road[id].maxSpeed = v[2];
        road[id].laneNumber = v[3];
        road[id].startId = v[4];
        road[id].endId = v[5];
        road[id].isDoubleSide = v[6];
        cnt++;
        //G[edge[v[0]][0]][edge[v[0]][1]] = v[0];  发现不用这么存图，路口相当于是链表存储边
        //G[edge[v[0]][1]][edge[v[0]][0]] = v[0];
        //cout << edge[v[0]].size() << endl;  都是2吗
    }
    cout << "Number of road = " << cnt << endl;
    fclose(F_road);
    return true;
}

bool ReadFile_car()
{
    string carPath = folder + "/car.txt";     // 10241
    int cnt = 0;
    FILE* F_car = fopen(carPath.data(), "r");
    if (!F_car) {
        cout << "The file" << carPath << "not exist" << endl;
        return 0;
    }
    while(!feof(F_car)) {
        char str[1005];
        //fscanf(F_car, "%s\n", str);  // 遇到第一个空格字符时，它会停止读取
        fgets(str, 1005, (FILE*)F_car);// 包括换行符
        //cout << str << endl;
        if (str[0] == '#') continue;
        vector<int> v;    // #(id,始发地,目的地,最高速度,出发时间)
        Split(str, v);
        /*
        cout << str << endl;
        for (int i: v) {
            cout << i << ' ';
        }
        cout << endl;
        */
        int id = v[0];
        car[id].id = v[0];
        car[id].startLoacation = v[1];
        car[id].endLocation = v[2];
        car[id].maxSpeed = v[3];
        car[id].startTime = v[4];
        cnt++;
    }
    cout << cnt << endl;
    fclose(F_car);
    return true;
}

// 路口
int maxVertex = 0;
bool ReadFile_cross()
{
    string cross = folder + "/cross.txt"; // 65点
    int cnt = 0;
    FILE* F_cross = fopen(cross.data(), "r");
    if (!F_cross) {
        cout << "The file" << cross << "not exist" << endl;
        return 0;
    }
    int maxID = -1, minID = INT_MAX;
    while(!feof(F_cross)) {
        char str[1005];
        //fscanf(F_cross, "%s\n", str);  // 遇到第一个空格字符时，它会停止读取
        fgets(str, 1005, (FILE*)F_cross);// 包括换行符
        if (str[0] == '#') continue;
        vector<int> v;    // #(结点id,道路id,道路id,道路id,道路id)一定是四个，但-1为空
        Split(str, v);
        cnt++;
        if (v[0] > maxID) maxID = v[0];
        if (v[0] < minID) minID = v[0];
        V.insert(v[0]);
        for(int i = 1; i <= 4; i++) {
            if (v[i] != -1) edge[v[i]].push_back(v[0]);
        }
    }

    cout << "Number of cross = " << cnt << endl;
    printf("minID = %d, maxID = %d\n", minID, maxID);
    maxVertex = maxID;
    fclose(F_cross);
    return true;
}

void Init()
{
    ReadFile_cross();  // 先路口后道路
    ReadFile_road();
    ReadFile_car();
    return;
}

void Test()
{
    std::vector<int> v;
    v.push_back(5);
    v.push_back(2);
    v.push_back(4);
    v.push_back(3);
    v.push_back(1);
    cout << v.size() << endl;
    cout << v[4] << endl;

    cout << "Chinese" << endl;
    cout << "中文" << endl;
    return;
}

const int maxn = 1e3 + 5;
const int inf = INT_MAX / 2;   // 要进行加法，不宜取最大值
int dis[maxn][maxn];
void Floyd(int n)    // 节点ID最大值
{
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            dis[i][j] = inf;
        }
        dis[i][i] = 0;
    }
    for (int i : E) {
        dis[edge[i][0]][edge[i][1]] = road[i].length;
        dis[edge[i][1]][edge[i][0]] = road[i].length;
    }
    for (int u = 1; u <= n; u++) {
        for (int v = 1; v <= n; v++) {
            for (int k = 1; k <= n; k++) {
               dis[u][v] = min(dis[u][v], dis[u][k] + dis[k][v]);
            }
        }
    }
    return;
}

void cal()
{

    return;
}


int main()
{
    Init();
    //Test();
    Floyd(maxVertex);
    cout << dis[2][5] << endl;
    return 0;
}
