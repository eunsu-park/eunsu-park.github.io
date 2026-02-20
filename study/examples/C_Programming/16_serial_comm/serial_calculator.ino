// serial_calculator.ino
// 시리얼 모니터 계산기

void setup() {
    Serial.begin(9600);

    Serial.println("=================================");
    Serial.println("   Simple Serial Calculator");
    Serial.println("=================================");
    Serial.println("Enter expression (e.g., 10 + 5)");
    Serial.println("Operators: +, -, *, /, %");
    Serial.println("Type 'quit' to exit");
    Serial.println("---------------------------------");
}

float calculate(float a, char op, float b) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/':
            if (b == 0) {
                Serial.println("Error: Division by zero");
                return 0;
            }
            return a / b;
        case '%':
            return (int)a % (int)b;
        default:
            Serial.print("Unknown operator: ");
            Serial.println(op);
            return 0;
    }
}

void processExpression(char* expr) {
    float num1, num2;
    char op;

    // 수식 파싱: "num1 op num2"
    int parsed = sscanf(expr, "%f %c %f", &num1, &op, &num2);

    if (parsed == 3) {
        float result = calculate(num1, op, num2);

        Serial.print(num1);
        Serial.print(" ");
        Serial.print(op);
        Serial.print(" ");
        Serial.print(num2);
        Serial.print(" = ");
        Serial.println(result);
    } else {
        Serial.println("Invalid format. Use: num1 op num2");
    }
}

char inputBuffer[32];
int inputIndex = 0;

void loop() {
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (inputIndex > 0) {
                inputBuffer[inputIndex] = '\0';

                // 종료 명령 확인
                if (strcmp(inputBuffer, "quit") == 0) {
                    Serial.println("Goodbye!");
                    while (1);  // 정지
                }

                processExpression(inputBuffer);
                inputIndex = 0;

                Serial.println("---------------------------------");
            }
        } else if (inputIndex < 31) {
            inputBuffer[inputIndex++] = c;
        }
    }
}

/*
 * 사용 방법:
 * 1. 시리얼 모니터 열기 (Tools → Serial Monitor)
 * 2. Baud rate를 9600으로 설정
 * 3. 수식 입력: "10 + 5" 엔터
 * 4. 결과 확인: "10 + 5 = 15.00"
 *
 * Wokwi에서:
 * - Serial Monitor 탭 클릭
 * - 수식 입력 후 엔터
 */
