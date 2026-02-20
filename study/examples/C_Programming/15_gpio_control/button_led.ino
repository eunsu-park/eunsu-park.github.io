// button_led.ino
// 버튼으로 LED 제어하기

const int BUTTON_PIN = 2;  // 버튼 핀
const int LED_PIN = 13;    // LED 핀

bool ledState = false;
bool lastButtonState = HIGH;

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);  // 내부 풀업 사용
    pinMode(LED_PIN, OUTPUT);

    Serial.begin(9600);
    Serial.println("Button LED Control");
    Serial.println("Press button to toggle LED");
}

void loop() {
    bool currentButtonState = digitalRead(BUTTON_PIN);

    // 버튼이 눌렸을 때 (HIGH → LOW)
    if (lastButtonState == HIGH && currentButtonState == LOW) {
        ledState = !ledState;  // LED 상태 토글
        digitalWrite(LED_PIN, ledState);

        Serial.print("LED ");
        Serial.println(ledState ? "ON" : "OFF");
    }

    lastButtonState = currentButtonState;
    delay(50);  // 간단한 디바운싱
}

/*
 * Wokwi 회로 구성:
 * 1. Arduino Uno 추가
 * 2. Pushbutton 추가
 * 3. 연결:
 *    - 버튼 한쪽 → 핀 2
 *    - 버튼 다른쪽 → GND
 *
 * 버튼을 누를 때마다 LED가 켜지고 꺼집니다.
 */
