// blink.ino
// LED 깜빡이기 - 가장 기본적인 Arduino 프로그램

const int LED_PIN = 13;  // 내장 LED (또는 LED_BUILTIN 사용)

void setup() {
    // 핀 모드를 출력으로 설정
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // LED 켜기
    digitalWrite(LED_PIN, HIGH);
    delay(1000);  // 1초 대기

    // LED 끄기
    digitalWrite(LED_PIN, LOW);
    delay(1000);  // 1초 대기
}

/*
 * Wokwi에서 실행:
 * 1. https://wokwi.com 접속
 * 2. New Project → Arduino Uno
 * 3. 이 코드 붙여넣기
 * 4. Start Simulation
 *
 * 내장 LED가 1초 간격으로 깜빡입니다.
 */
