/**
 * Bank Authentication Keys
 * Hardcoded 64-bit keys for major Korean banks
 */

// Bank Auth Keys - 64-bit unique identifiers
export const BANK_AUTH_KEYS = {
    KB: {
        id: 'KB',
        name: 'KB국민은행',
        nameEn: 'Kookmin Bank',
        key: BigInt('0x4B42524F4F4B4D4E'), // "KBROOKMN"
        color: '#FFB300', // Yellow
    },
    SHINHAN: {
        id: 'SHINHAN',
        name: '신한은행',
        nameEn: 'Shinhan Bank',
        key: BigInt('0x5348494E48414E42'), // "SHINHANB"
        color: '#0046FF', // Blue
    },
    WOORI: {
        id: 'WOORI',
        name: '우리은행',
        nameEn: 'Woori Bank',
        key: BigInt('0x574F4F5249424E4B'), // "WOORIBNK"
        color: '#0066B3', // Blue
    },
    TOSS: {
        id: 'TOSS',
        name: '토스뱅크',
        nameEn: 'Toss Bank',
        key: BigInt('0x544F535342414E4B'), // "TOSSBANK"
        color: '#0064FF', // Toss Blue
    },
    HANA: {
        id: 'HANA',
        name: '하나은행',
        nameEn: 'Hana Bank',
        key: BigInt('0x48414E4142414E4B'), // "HANABANK"
        color: '#009775', // Green
    },
    NH: {
        id: 'NH',
        name: 'NH농협은행',
        nameEn: 'Nonghyup Bank',
        key: BigInt('0x4E4842414E4B4B52'), // "NHBANKKR"
        color: '#00BE00', // NH Green
    },
};

// Get bank list for dropdown/picker
export const BANK_LIST = Object.values(BANK_AUTH_KEYS);

// Lookup bank by 64-bit key (for decoding)
export const getBankByKey = (keyBigInt) => {
    for (const bank of BANK_LIST) {
        if (bank.key === keyBigInt) {
            return bank;
        }
    }
    return null; // Unknown bank
};

// Lookup bank by ID
export const getBankById = (id) => {
    return BANK_AUTH_KEYS[id] || null;
};
