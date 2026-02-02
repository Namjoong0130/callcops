import React, { useState, useRef } from 'react';
import { StyleSheet, View, Text, ActivityIndicator, SafeAreaView, StatusBar, Platform, BackHandler } from 'react-native';
import { WebView } from 'react-native-webview';
import { useEffect } from 'react';

// Configure your web app URL here
// For development: use your Mac's local IP (e.g., http://192.168.x.x:5173)
// For production: use a deployed URL
const WEB_APP_URL = 'http://10.0.2.2:5173/phone'; // Phone simulator page

export default function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const webViewRef = useRef(null);

  // Handle Android back button
  useEffect(() => {
    if (Platform.OS === 'android') {
      const backHandler = BackHandler.addEventListener('hardwareBackPress', () => {
        if (webViewRef.current) {
          webViewRef.current.goBack();
          return true;
        }
        return false;
      });
      return () => backHandler.remove();
    }
  }, []);

  const handleLoadEnd = () => {
    setIsLoading(false);
  };

  const handleError = (syntheticEvent) => {
    const { nativeEvent } = syntheticEvent;
    setError(nativeEvent.description || 'Failed to load web app');
    setIsLoading(false);
  };

  if (error) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#111827" />
        <View style={styles.errorContainer}>
          <Text style={styles.errorTitle}>연결 실패</Text>
          <Text style={styles.errorText}>{error}</Text>
          <Text style={styles.helpText}>
            1. callcops-frontend 서버가 실행 중인지 확인하세요{'\n'}
            2. 같은 Wi-Fi 네트워크에 연결되어 있는지 확인하세요{'\n'}
            3. App.js의 WEB_APP_URL이 올바른지 확인하세요
          </Text>
          <Text style={styles.urlText}>현재 URL: {WEB_APP_URL}</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#111827" />

      <WebView
        ref={webViewRef}
        source={{ uri: WEB_APP_URL }}
        style={styles.webview}
        onLoadEnd={handleLoadEnd}
        onError={handleError}
        allowsInlineMediaPlayback={true}
        mediaPlaybackRequiresUserAction={false}
        javaScriptEnabled={true}
        domStorageEnabled={true}
        startInLoadingState={true}
        // Allow microphone access
        mediaCapturePermissionGrantType="grant"
        allowsBackForwardNavigationGestures={true}
        // Inject CSS to ensure mobile-friendly display
        injectedJavaScript={`
          const meta = document.createElement('meta');
          meta.setAttribute('name', 'viewport');
          meta.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no');
          document.head.appendChild(meta);
          true;
        `}
      />

      {isLoading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#3b82f6" />
          <Text style={styles.loadingText}>CallCops 로딩 중...</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
  },
  webview: {
    flex: 1,
    backgroundColor: '#111827',
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#111827',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#9ca3af',
    marginTop: 16,
    fontSize: 16,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ef4444',
    marginBottom: 12,
  },
  errorText: {
    fontSize: 16,
    color: '#9ca3af',
    textAlign: 'center',
    marginBottom: 24,
  },
  helpText: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'left',
    lineHeight: 24,
    marginBottom: 16,
  },
  urlText: {
    fontSize: 12,
    color: '#4b5563',
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
  },
});
