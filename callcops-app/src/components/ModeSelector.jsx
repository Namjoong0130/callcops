/**
 * ModeSelector - Main screen for choosing sender/receiver mode
 */
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

export default function ModeSelector({ onSelectMode }) {
    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#7c2d12', '#831843', '#2e1065']} // Call Screen Theme
                style={StyleSheet.absoluteFill}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
            />

            {/* Header */}
            <View style={styles.header}>
                <View style={styles.iconCircle}>
                    <Ionicons name="shield-checkmark" size={40} color="#fff" />
                </View>
                <Text style={styles.title}>CallCops</Text>
                <Text style={styles.subtitle}>통화 인증 시스템</Text>
            </View>

            {/* Mode Selection */}
            <View style={styles.buttons}>
                {/* Sender Mode */}
                <TouchableOpacity
                    style={[styles.modeButton, styles.senderButton]}
                    onPress={() => onSelectMode('sender')}
                    activeOpacity={0.8}
                >
                    <View style={[styles.modeIcon, styles.senderIcon]}>
                        <Ionicons name="mic" size={28} color="#fff" />
                    </View>
                    <View style={styles.modeInfo}>
                        <Text style={styles.modeTitle}>송신자 모드</Text>
                        <Text style={styles.modeDesc}>음성 녹음 → 워터마크 삽입 → 다운로드</Text>
                    </View>
                </TouchableOpacity>

                {/* Receiver Mode */}
                <TouchableOpacity
                    style={[styles.modeButton, styles.receiverButton]}
                    onPress={() => onSelectMode('receiver')}
                    activeOpacity={0.8}
                >
                    <View style={[styles.modeIcon, styles.receiverIcon]}>
                        <Ionicons name="call" size={28} color="#fff" />
                    </View>
                    <View style={styles.modeInfo}>
                        <Text style={styles.modeTitle}>수신자 모드</Text>
                        <Text style={styles.modeDesc}>파일 업로드 → 워터마크 검증 → 발신자 확인</Text>
                    </View>
                </TouchableOpacity>
            </View>

            {/* Footer */}
            <Text style={styles.footer}>128-bit Watermark • CRC-16 Verification</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#111827',
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 24,
    },
    header: {
        alignItems: 'center',
        marginBottom: 48,
    },
    iconCircle: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: 'rgba(255, 255, 255, 0.1)', // Transparent White
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.2)',
    },
    title: {
        fontSize: 32,
        fontWeight: 'bold',
        color: '#fff',
        marginBottom: 8,
    },
    subtitle: {
        fontSize: 16,
        color: '#d1d5db', // Lighter Gray
    },
    buttons: {
        width: '100%',
        gap: 16,
    },
    modeButton: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 20,
        borderRadius: 16,
        borderWidth: 1,
    },
    senderButton: {
        backgroundColor: 'rgba(190, 24, 93, 0.1)', // Pink theme (matched to border)
        borderColor: '#be185d', // Dark Pink Border
    },
    receiverButton: {
        backgroundColor: 'rgba(96, 165, 250, 0.1)', // Brighter Blue (400)
        borderColor: 'rgba(255, 255, 255, 0.4)', // White Border
    },
    modeIcon: {
        width: 56,
        height: 56,
        borderRadius: 28,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 16,
    },
    senderIcon: {
        backgroundColor: 'rgba(190, 24, 93, 0.2)', // Pink theme
    },
    receiverIcon: {
        backgroundColor: 'rgba(96, 165, 250, 0.2)', // Brighter Blue
    },
    modeIconText: {
        fontSize: 24,
    },
    modeInfo: {
        flex: 1,
    },
    modeTitle: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#fff',
        marginBottom: 4,
    },
    modeDesc: {
        fontSize: 12, // Reduced size
        color: '#e5e7eb', // Lighter Gray
    },
    // Arrow style removed
    footer: {
        marginTop: 48,
        fontSize: 12,
        color: '#d1d5db', // Lighter Gray (Gray-300)
    },
});
