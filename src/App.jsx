import './App.css'
import Homepage from './Profile/Homepage.jsx'
import AOS from 'aos';
import 'aos/dist/aos.css';
AOS.init();


function App(props) {
  console.log(props)
  return (
    <Homepage />
  )
}

export default App;
