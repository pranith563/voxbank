import './App.css';
import VoiceSearch from './VoiceSearch';

function App() {
  return (
    <div className="App min-h-screen flex flex-col items-center justify-center">
      <header className="App-header w-full text-center">
        <h1>VoxBank</h1>
        <p>Your AI Voice Banking Companion</p>
      </header>
      <VoiceSearch/>
    </div>
  );
}

export default App;

