from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

PEOPLE = ["Kanokphan", "Yensa"]
BANK_FOLDER = "BankAccount"
CC_FOLDER = "CreditCard"

PERSON_COLORS = {
    "Kanokphan": "#378ADD",
    "Yensa": "#1D9E75",
}

CC_CATEGORIES = {
    "Groceries": [
        "lotus", "tops", "golden place", "mall sup", "the mall sup",
        "makro", "bigc", "big c", "foodland", "villa market",
    ],
    "Food & Dining": [
        "shabushi", "saemaeul", "mo-mo paradise", "after you", "thongsmith",
        "restaurant", "s&p", "paradise", "omakase", "sizzler", "pizza", "burger",
        "kfc", "mcdonald", "bakery", "noodle", "ramen", "sushi", "bbq",
        "go heang", "สมชาย", "ข้าว", "อาหาร",
    ],
    "Fuel": ["bangchak", "pttst", "ptt ", "shell", "caltex", "esso", "ปตท"],
    "Health & Fitness": [
        "fitwhey", "fitness", "gym", "24seven", "sport", "protein",
        "supplement", "yoga", "pilates", "crossfit",
    ],
    "Healthcare": [
        "hospital", "clinic", "vejthani", "pharmacy", "drug",
        "medical", "dental", "eye", "โรงพยาบาล", "คลินิก",
    ],
    "Home & Hardware": [
        "home product", "hardware", "thai watsadu", "index", "ikea",
        "dcasa", "homepro", "boonthavorn",
    ],
    "Entertainment": [
        "major", "cinema", "movie", "netflix", "spotify",
        "youtube", "disney", "game", "steam",
    ],
    "Tech & Digital": [
        "runpod", "financial times", "aws", "google", "apple",
        "microsoft", "adobe", "dropbox", "chatgpt", "openai",
        "anthropic", "amz", "amazon",
    ],
    "Beverages & Cafe": [
        "cfw", "peak chocolate", "chao phraya", "cha tra mue",
        "black canyon", "inthanin", "doi chaang", "coffee world",
        "starbucks", "cafe amazon",
    ],
    "Shopping & Fashion": [
        "shopee", "lazada", "daisen", "central", "robinson",
        "zara", "h&m", "uniqlo", "adidas", "nike", "(for shopee)",
    ],
    "Education": ["kasetsart", "university", "course", "udemy", "coursera"],
    "Transport": ["grab", "bolt", "taxi", "bts", "mrt", "airport rail", "แกร็บ"],
    "Utilities": ["true", "ais", "dtac", "ntnt", "electricity", "water"],
    "Travel": [
        "airline", "thai airways", "airasia", "nok air",
        "hotel", "booking", "agoda", "airbnb",
    ],
}

BANK_CATEGORIES = {
    "Investment": [
        "securities", "kasikorn securi", "cimbt", "direct debit",
        "ksecurities", "หลักทรัพย์",
    ],
    "Education": ["kasetsart", "university", "มหาวิทยาลัย", "เกษตรศาสตร์"],
    "Healthcare": ["clinic", "hospital", "vejthani", "chamnan"],
    "E-commerce": ["shopee", "lazada", "amazon"],
    "Regular Fixed Transfer": ["sungsud"],
    "Family / Personal": [
        "woraya", "saijai", "kunlayawadee", "pornparn",
        "kanyarat", "suwit", "nunchu",
    ],
    "Food & Dining": [
        "go heang", "omakase", "paradise", "มั่งมี",
        "shabushi", "ร้านอาหาร",
    ],
    "Beverages & Cafe": ["ชอบชา", "coffee", "cafe amazon", "starbucks"],
    "Health & Fitness": ["fitwhey", "fitness", "24seven"],
    "Fuel": ["ptt", "bangchak", "synergy", "caltex"],
    "Groceries": ["mall", "tops", "lotus", "golden place", "the mall sup"],
    "Cash Withdrawal": ["cash withdrawal", "atm"],
    "Interest": ["interest deposit", "interest"],
    "Incoming Transfer": ["transfer deposit"],
}

CATEGORY_COLORS = {
    "Groceries": "#1D9E75",
    "Food & Dining": "#378ADD",
    "Fuel": "#EF9F27",
    "Health & Fitness": "#D85A30",
    "Healthcare": "#7F77DD",
    "Home & Hardware": "#0F6E56",
    "Entertainment": "#D4537E",
    "Tech & Digital": "#534AB7",
    "Beverages & Cafe": "#BA7517",
    "Shopping & Fashion": "#185FA5",
    "Education": "#639922",
    "Transport": "#5F5E5A",
    "Investment": "#A32D2D",
    "Family / Personal": "#712B13",
    "Regular Fixed Transfer": "#B4B2A9",
    "Cash Withdrawal": "#888780",
    "Interest": "#27500A",
    "Incoming Transfer": "#3B6D11",
    "E-commerce": "#7F77DD",
    "Utilities": "#63381F",
    "Travel": "#3C3489",
    "Other": "#D3D1C7",
}

EXCLUDE_FROM_LIFESTYLE = ["Investment", "Incoming Transfer", "Interest"]
