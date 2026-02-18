int gcd(int a, int b) {
  if (a == 0)
    return b;
  while (b != 0) {
    if (a > b)
      a = a - b;
    else
      b = b - a;
  }
  return a;
}

int main() {
  int a = *(volatile int*)4;
  int b = *(volatile int*)8;
  int c = gcd(a, b);
  *(volatile int*)12 = c;
  return 0;
}
