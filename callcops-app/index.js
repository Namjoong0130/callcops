import { registerRootComponent } from 'expo';

// Polyfill Buffer for React Native
global.Buffer = global.Buffer || require('buffer').Buffer;

import App from './App';

// registerRootComponent calls AppRegistry.registerComponent('main', () => App);
// It also ensures that whether you load the app in Expo Go or in a native build,
// the environment is set up appropriately
registerRootComponent(App);
