#include <iostream>
using namespace std;
int main(){
  int N;
  cin >> N;
  int first_max_num = 0;
  int first_max_idx = 0;
  int second_max_num = 0;
  int second_max_idx = 0;
  for(int i = 1; i <= N; i++){
    int A;
    cin >> A;
    //��i������Ŀǰ�x������Ĕ�߀��r
    //ԭ������Ĕ�׃�ɵڶ����,�µĔ�׃�����Ĕ�
    if(A > first_max_num){
      second_max_num = first_max_num;
      second_max_idx = first_max_idx;
      first_max_num = A;
      first_max_idx = i;
    }else if(A > second_max_num){
      //�x��Ĕ��]�д�����ģ�����춵ڶ����
      second_max_num = A;
      second_max_idx = i;
    }
  }
  cout << second_max_idx << endl;
  return 0;
}