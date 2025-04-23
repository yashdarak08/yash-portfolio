import React from 'react';
import { Link } from 'react-router-dom';
import './PassionPage.css';
import { FaArrowLeft } from 'react-icons/fa';

export default function PassionPage() {
  document.title = "Yash Darak | Passions";
  return (
    <div className="passion-container">
      <div className="passion-header">
        <h1 className="passion-title">My Passions</h1>
      </div>
      
      <div className="passion-content">
        <section className="passion-section">
          <h2>Sudoku Championships</h2>
          <p>
            My journey with Sudoku began in middle school when I first discovered the logic puzzle 
            in a newspaper. What started as a casual interest quickly developed into a passion for 
            competitive solving.
          </p>
          <p>
            I've competed in multiple Indian Sudoku Championships, consistently ranking among 
            the top performers. The mental discipline and pattern recognition skills I've developed
            through Sudoku competitions have deeply influenced my approach to problem-solving in
            mathematics and computer science.
          </p>
          <div className="passion-highlight">
            "Logic puzzles like Sudoku train the mind to break complex problems into simpler patterns - 
            a skill that translates perfectly to algorithm design and mathematical analysis."
          </div>
        </section>

        <section className="passion-section">
          <h2>Sports: Football & Cricket</h2>
          <p>
            Beyond academics and professional pursuits, I maintain an active lifestyle through my love
            for football and cricket. These team sports have taught me valuable lessons about collaboration,
            strategy, and performing under pressure.
          </p>
          <p>
            Playing midfielder in football has honed my ability to think several steps ahead, while cricket
            has taught me patience and the importance of technique. Both sports provide a balance to my
            technical work and keep me physically active.
          </p>
        </section>

        <section className="passion-section">
          <h2>ABACUS/UCMAS Gold Medalist</h2>
          <p>
            My early training in mental mathematics through ABACUS/UCMAS laid the foundation for my
            computational thinking. Earning a gold medal required thousands of hours of practice and
            developing the ability to perform complex calculations mentally at high speed.
          </p>
          <p>
            This mental mathematics training has given me a unique perspective on numbers and patterns,
            contributing significantly to my success in mathematics, algorithm design, and machine learning work.
          </p>
          <p>
            The mental discipline acquired through ABACUS training continues to influence how I approach
            complex problems, breaking them down into manageable components that can be solved systematically.
          </p>
        </section>
      </div>
    </div>
  );
}