-- MySQL dump 10.13  Distrib 8.0.16, for Win64 (x86_64)
--
-- Host: localhost    Database: bowling
-- ------------------------------------------------------
-- Server version	8.0.16

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
 SET NAMES utf8 ;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `player`
--

DROP TABLE IF EXISTS `player`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `player` (
  `idplayer` int(11) NOT NULL AUTO_INCREMENT,
  `game_idgame` int(11) NOT NULL,
  `player_score` int(10) unsigned DEFAULT NULL,
  `player_name` varchar(25) COLLATE utf8_turkish_ci DEFAULT NULL,
  `winner` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`idplayer`),
  KEY `fk_player_game_idx` (`game_idgame`),
  CONSTRAINT `fk_player_game` FOREIGN KEY (`game_idgame`) REFERENCES `game` (`idgame`)
) ENGINE=InnoDB AUTO_INCREMENT=25 DEFAULT CHARSET=utf8 COLLATE=utf8_turkish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `player`
--

LOCK TABLES `player` WRITE;
/*!40000 ALTER TABLE `player` DISABLE KEYS */;
INSERT INTO `player` VALUES (1,1,NULL,'Eric Pennington',NULL),(2,1,NULL,'Abdul Compton',NULL),(3,2,101,'Umer Dowling',0),(4,2,69,'Isla-Rae Clarke',1),(5,3,NULL,'Justin Coombes',NULL),(6,4,NULL,'Constance Finley',NULL),(7,4,NULL,'Cheyanne O\'Ryan',NULL),(8,4,NULL,'Augustus Dickinson',NULL),(9,4,NULL,'Shauna Dorsey',NULL),(10,5,NULL,'Susannah Kramer',NULL),(11,5,NULL,'Betty Li',NULL),(12,6,NULL,'Raees Snyder',NULL),(13,7,NULL,'Jim Ramos',NULL),(14,7,NULL,'Adnan Olson',NULL),(15,7,NULL,'Muhammed Goulding',NULL),(16,7,NULL,'Shawn Brookes',NULL),(17,7,NULL,'Cooper Collier',NULL),(18,8,NULL,'Dante Pineda',NULL),(19,8,NULL,'Hakeem Clifford',NULL),(20,8,NULL,'Vinay Dominguez',NULL),(21,9,NULL,'Rahma Molina',NULL),(22,9,NULL,'Kirk Jordan',NULL),(23,10,NULL,'Fathima Adamson',NULL),(24,10,NULL,'Lilli Wilcox',NULL);
/*!40000 ALTER TABLE `player` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-07-28  0:23:58
