/**
 * ModeSelector - Main screen for choosing sender/receiver mode
 */
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

export default function ModeSelector({ onSelectMode }) {
    return (
        <View style={styles.container}>
            {/* Header */}
            <View style={styles.header}>
                <View style={styles.iconCircle}>
                    <Text style={styles.iconText}>📞</Text>
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
                        <Text style={styles.modeIconText}>✨</Text>
                    </View>
                    <View style={styles.modeInfo}>
                        <Text style={styles.modeTitle}>송신자 모드</Text>
                        <Text style={styles.modeDesc}>음성 녹음 → 워터마크 삽입 → 다운로드</Text>
                    </View>
                    <Text style={styles.arrow}>›</Text>
                </TouchableOpacity>

                {/* Receiver Mode */}
                <TouchableOpacity
                    style={[styles.modeButton, styles.receiverButton]}
                    onPress={() => onSelectMode('receiver')}
                    activeOpacity={0.8}
                >
                    <View style={[styles.modeIcon, styles.receiverIcon]}>
                        <Text style={styles.modeIconText}>🛡️</Text>
                    </View>
                    <View style={styles.modeInfo}>
                        <Text style={styles.modeTitle}>수신자 모드</Text>
                        <Text style={styles.modeDesc}>파일 업로드 → 워터마크 검증 → 발신자 확인</Text>
                    </View>
                    <Text style={styles.arrow}>›</Text>
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
        backgroundColor: '#3b82f6',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
    },
    iconText: {
        fontSize: 36,
    },
    title: {
        fontSize: 32,
        fontWeight: 'bold',
        color: '#fff',
        marginBottom: 8,
    },
    subtitle: {
        fontSize: 16,
        color: '#9ca3af',
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
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        borderColor: 'rgba(34, 197, 94, 0.3)',
    },
    receiverButton: {
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderColor: 'rgba(59, 130, 246, 0.3)',
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
        backgroundColor: 'rgba(34, 197, 94, 0.2)',
    },
    receiverIcon: {
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
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
        fontSize: 12,
        color: '#9ca3af',
    },
    arrow: {
        fontSize: 24,
        color: '#6b7280',
    },
    footer: {
        marginTop: 48,
        fontSize: 12,
        color: '#6b7280',
    },
});
