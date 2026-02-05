/**
 * CallCops - Standalone Native App
 * No WebView required - all logic runs natively
 */
import React, { useState, useEffect } from 'react';
import { StyleSheet, View, StatusBar, PermissionsAndroid, Platform } from 'react-native';
import ModeSelector from './src/components/ModeSelector';
import SenderMode from './src/components/SenderMode';
import ReceiverMode from './src/components/ReceiverMode';

export default function App() {
  const [mode, setMode] = useState('select'); // select, sender, receiver

  // Request permissions on mount (Android)
  useEffect(() => {
    const requestPermissions = async () => {
      if (Platform.OS === 'android') {
        try {
          await PermissionsAndroid.requestMultiple([
            PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
            PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE,
          ]);
        } catch (err) {
          console.warn('Permission request failed:', err);
        }
      }
    };
    requestPermissions();
  }, []);

  const handleSelectMode = (selectedMode) => {
    setMode(selectedMode);
  };

  const handleBack = () => {
    setMode('select');
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#111827" />

      {mode === 'select' && (
        <ModeSelector onSelectMode={handleSelectMode} />
      )}

      {mode === 'sender' && (
        <SenderMode onBack={handleBack} />
      )}

      {mode === 'receiver' && (
        <ReceiverMode onBack={handleBack} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
  },
});
