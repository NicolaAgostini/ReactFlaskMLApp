import React from 'react';


const Navbar = () => {
  return (
    <nav className="navbar">
      <h1>Artificial Intelligence Blog</h1>
      <div className="links">
        <a href="/binary_classification">Binary Classification</a>
        <a href="/gan">GAN</a>
      </div>
    </nav>
  );
}
 
export default Navbar;