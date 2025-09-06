USE workshop;
CREATE TABLE Filmes (
  codeID VARCHAR(15) NOT NULL,
  name VARCHAR(100) NOT NULL,
  type VARCHAR(45) NOT NULL,
  year YEAR NOT NULL,
  PRIMARY KEY (codeID)
);

INSERT INTO Filmes (codeID, name, type, year)
VALUES
('tt1517268', 'Barbie', 'Comédia', 2023);
('tt4154796', 'Vingadores: Ultimato', 'Ação', 2019),
('tt1375666', 'A Origem', 'Ficção Científica', 2010),
('tt0110912', 'Pulp Fiction', 'Crime', 1994),
('tt0133093', 'Matrix', 'Ficção Científica', 1999),
('tt6751668', 'Parasita', 'Drama', 2019),
('tt0120737', 'O Senhor dos Anéis: A Sociedade do Anel', 'Fantasia', 2001),
('tt0068646', 'O Poderoso Chefão', 'Crime', 1972),
('tt4154756', 'Vingadores: Guerra Infinita', 'Ação', 2018),
('tt0468569', 'Batman: O Cavaleiro das Trevas', 'Ação', 2008);

select * FROM Filmes; 

