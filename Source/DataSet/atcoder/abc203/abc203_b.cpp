#define _GLIBCXX_DEBUG
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define rep(i,a,b) for(int i=a;i<b;i++)
using vi = vector<int>; // int��1��Ԫ���ͤ� vi �Ȥ����e����Ĥ���
using vvi = vector<vi>; // int��2��Ԫ���ͤ� vvi �Ȥ����e����Ĥ���
int main() {
  int n,k;
  cin>>n>>k;
  cout<<100*k*n*(n+1)/2+n*k*(k+1)/2;
}