/*
 * Weather App
 * OpenWeatherMap API를 사용한 날씨 앱
 *
 * 주의: 실제 서비스에서는 API 키를 환경 변수나 서버에서 관리해야 합니다.
 * 이 예제는 학습용으로 무료 API를 사용합니다.
 */

// ============================================
// 설정
// ============================================
// 무료 데모 API (실제 서비스에서는 자체 키 필요)
// https://openweathermap.org/api 에서 무료 키 발급 가능
const API_KEY = 'demo'; // 실제 키로 교체 필요
const API_BASE = 'https://api.openweathermap.org/data/2.5/weather';

// 데모 모드 (API 키가 없을 때 샘플 데이터 사용)
const DEMO_MODE = API_KEY === 'demo';

// ============================================
// DOM Elements
// ============================================
const cityInput = document.getElementById('cityInput');
const searchBtn = document.getElementById('searchBtn');
const quickCityBtns = document.querySelectorAll('.city-btn');
const loadingEl = document.getElementById('loading');
const errorEl = document.getElementById('error');
const errorMessageEl = document.getElementById('errorMessage');
const weatherDisplay = document.getElementById('weatherDisplay');

// ============================================
// 샘플 데이터 (데모용)
// ============================================
const sampleWeatherData = {
    'Seoul': {
        name: 'Seoul',
        sys: { country: 'KR', sunrise: 1706400000, sunset: 1706436000 },
        main: { temp: 3, feels_like: -1, humidity: 45, pressure: 1020 },
        weather: [{ main: 'Clear', description: '맑음', icon: '01d' }],
        wind: { speed: 2.5 },
        visibility: 10000,
        clouds: { all: 10 }
    },
    'Tokyo': {
        name: 'Tokyo',
        sys: { country: 'JP', sunrise: 1706396400, sunset: 1706432400 },
        main: { temp: 8, feels_like: 5, humidity: 55, pressure: 1015 },
        weather: [{ main: 'Clouds', description: '구름 조금', icon: '02d' }],
        wind: { speed: 3.1 },
        visibility: 8000,
        clouds: { all: 25 }
    },
    'New York': {
        name: 'New York',
        sys: { country: 'US', sunrise: 1706443200, sunset: 1706479200 },
        main: { temp: -2, feels_like: -7, humidity: 60, pressure: 1008 },
        weather: [{ main: 'Snow', description: '눈', icon: '13d' }],
        wind: { speed: 5.2 },
        visibility: 3000,
        clouds: { all: 90 }
    },
    'London': {
        name: 'London',
        sys: { country: 'GB', sunrise: 1706428800, sunset: 1706461200 },
        main: { temp: 6, feels_like: 3, humidity: 80, pressure: 1012 },
        weather: [{ main: 'Rain', description: '비', icon: '10d' }],
        wind: { speed: 4.1 },
        visibility: 6000,
        clouds: { all: 75 }
    },
    'Paris': {
        name: 'Paris',
        sys: { country: 'FR', sunrise: 1706425200, sunset: 1706458800 },
        main: { temp: 5, feels_like: 2, humidity: 70, pressure: 1010 },
        weather: [{ main: 'Clouds', description: '흐림', icon: '04d' }],
        wind: { speed: 3.5 },
        visibility: 7000,
        clouds: { all: 65 }
    }
};

// ============================================
// 초기화
// ============================================
function init() {
    addEventListeners();

    // 데모 모드 알림
    if (DEMO_MODE) {
        console.log('데모 모드: 샘플 데이터를 사용합니다.');
        console.log('실제 API를 사용하려면 app.js의 API_KEY를 설정하세요.');
    }

    // 초기 도시 로드
    fetchWeather('Seoul');
}

// ============================================
// 이벤트 리스너
// ============================================
function addEventListeners() {
    // 검색 버튼
    searchBtn.addEventListener('click', () => {
        const city = cityInput.value.trim();
        if (city) {
            fetchWeather(city);
        }
    });

    // Enter 키
    cityInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const city = cityInput.value.trim();
            if (city) {
                fetchWeather(city);
            }
        }
    });

    // 빠른 도시 버튼
    quickCityBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const city = btn.dataset.city;
            cityInput.value = city;
            fetchWeather(city);
        });
    });
}

// ============================================
// 날씨 데이터 가져오기
// ============================================
async function fetchWeather(city) {
    showLoading();
    hideError();
    hideWeather();

    try {
        let data;

        if (DEMO_MODE) {
            // 데모 모드: 샘플 데이터 사용
            await simulateDelay(800);
            data = getDemoData(city);
        } else {
            // 실제 API 호출
            const url = `${API_BASE}?q=${encodeURIComponent(city)}&appid=${API_KEY}&units=metric&lang=kr`;
            const response = await fetch(url);

            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error(`'${city}'를 찾을 수 없습니다.`);
                } else if (response.status === 401) {
                    throw new Error('API 키가 유효하지 않습니다.');
                } else {
                    throw new Error('날씨 정보를 가져오는데 실패했습니다.');
                }
            }

            data = await response.json();
        }

        displayWeather(data);
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// ============================================
// 데모 데이터 처리
// ============================================
function getDemoData(city) {
    // 정확한 이름 매칭
    if (sampleWeatherData[city]) {
        return sampleWeatherData[city];
    }

    // 대소문자 무시 매칭
    const cityLower = city.toLowerCase();
    for (const [key, value] of Object.entries(sampleWeatherData)) {
        if (key.toLowerCase() === cityLower) {
            return value;
        }
    }

    // 한글 도시명 매핑
    const koreanCities = {
        '서울': 'Seoul',
        '도쿄': 'Tokyo',
        '뉴욕': 'New York',
        '런던': 'London',
        '파리': 'Paris'
    };

    if (koreanCities[city]) {
        return sampleWeatherData[koreanCities[city]];
    }

    // 찾을 수 없음
    throw new Error(`'${city}'를 찾을 수 없습니다. 데모 모드에서는 Seoul, Tokyo, New York, London, Paris만 지원됩니다.`);
}

function simulateDelay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// 날씨 표시
// ============================================
function displayWeather(data) {
    // 도시 정보
    document.getElementById('cityName').textContent = data.name;
    document.getElementById('country').textContent = getCountryName(data.sys.country);

    // 온도
    document.getElementById('temp').textContent = Math.round(data.main.temp);

    // 날씨 아이콘
    const iconCode = data.weather[0].icon;
    const iconUrl = `https://openweathermap.org/img/wn/${iconCode}@2x.png`;
    document.getElementById('weatherIcon').src = iconUrl;
    document.getElementById('weatherIcon').alt = data.weather[0].description;

    // 설명
    document.getElementById('description').textContent = data.weather[0].description;

    // 상세 정보
    document.getElementById('feelsLike').textContent = `${Math.round(data.main.feels_like)}°C`;
    document.getElementById('humidity').textContent = `${data.main.humidity}%`;
    document.getElementById('windSpeed').textContent = `${data.wind.speed} m/s`;
    document.getElementById('pressure').textContent = `${data.main.pressure} hPa`;
    document.getElementById('visibility').textContent = `${(data.visibility / 1000).toFixed(1)} km`;
    document.getElementById('clouds').textContent = `${data.clouds.all}%`;

    // 일출/일몰
    document.getElementById('sunrise').textContent = formatTime(data.sys.sunrise);
    document.getElementById('sunset').textContent = formatTime(data.sys.sunset);

    // 업데이트 시간
    document.getElementById('updateTime').textContent = new Date().toLocaleTimeString('ko-KR');

    // 배경 변경 (날씨에 따라)
    updateBackground(data.weather[0].main);

    showWeather();
}

// ============================================
// 유틸리티 함수
// ============================================
function formatTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('ko-KR', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function getCountryName(code) {
    const countries = {
        'KR': '대한민국',
        'JP': '일본',
        'US': '미국',
        'GB': '영국',
        'FR': '프랑스',
        'CN': '중국',
        'DE': '독일',
        'IT': '이탈리아',
        'ES': '스페인',
        'AU': '호주'
    };
    return countries[code] || code;
}

function updateBackground(weatherMain) {
    const app = document.querySelector('.app');

    // 기존 클래스 제거
    app.classList.remove('sunny', 'cloudy', 'rainy', 'snowy');

    // 날씨에 따른 클래스 추가
    switch (weatherMain.toLowerCase()) {
        case 'clear':
            app.classList.add('sunny');
            break;
        case 'clouds':
            app.classList.add('cloudy');
            break;
        case 'rain':
        case 'drizzle':
        case 'thunderstorm':
            app.classList.add('rainy');
            break;
        case 'snow':
            app.classList.add('snowy');
            break;
    }
}

// ============================================
// UI 상태 관리
// ============================================
function showLoading() {
    loadingEl.classList.remove('hidden');
}

function hideLoading() {
    loadingEl.classList.add('hidden');
}

function showError(message) {
    errorMessageEl.textContent = message;
    errorEl.classList.remove('hidden');
}

function hideError() {
    errorEl.classList.add('hidden');
}

function showWeather() {
    weatherDisplay.classList.remove('hidden');
}

function hideWeather() {
    weatherDisplay.classList.add('hidden');
}

// ============================================
// 앱 시작
// ============================================
init();
