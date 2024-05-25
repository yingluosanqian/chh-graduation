#include <iostream>
using namespace std;

int bar(int x) {
  return x - 1;
}
void foo() {
  int y = bar(1);
  for (int i=0; i<y; i++)
    cout << "Never Reach.";
}
int main() {
  return 0;
}