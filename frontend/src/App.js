import React from 'react';
import './App.css';


import Navbar from './Navbar';
import Binary_Classification from './binary_classification';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <div className="content">
          <Switch>
            <Route path="/binary_classification">
              <Binary_Classification />
            </Route>
          </Switch>
        </div>
      </div>
    </Router>
  );
}

export default App;