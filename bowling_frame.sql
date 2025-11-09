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
-- Table structure for table `frame`
--

DROP TABLE IF EXISTS `frame`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
 SET character_set_client = utf8mb4 ;
CREATE TABLE `frame` (
  `idframe` int(11) NOT NULL AUTO_INCREMENT,
  `player_idplayer` int(11) NOT NULL,
  `frame_number` int(10) unsigned DEFAULT NULL,
  `shot1` int(10) unsigned DEFAULT NULL,
  `shot2` int(10) unsigned DEFAULT NULL,
  `shot3` int(10) unsigned DEFAULT NULL,
  `frame_score` int(11) DEFAULT NULL,
  `strike` tinyint(1) DEFAULT NULL,
  `spare` tinyint(1) DEFAULT NULL,
  `gutter` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`idframe`),
  KEY `fk_frame_player1_idx` (`player_idplayer`),
  CONSTRAINT `fk_frame_player1` FOREIGN KEY (`player_idplayer`) REFERENCES `player` (`idplayer`)
) ENGINE=InnoDB AUTO_INCREMENT=21 DEFAULT CHARSET=utf8 COLLATE=utf8_turkish_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `frame`
--

LOCK TABLES `frame` WRITE;
/*!40000 ALTER TABLE `frame` DISABLE KEYS */;
INSERT INTO `frame` VALUES (1,3,1,9,1,NULL,9,0,0,0),(2,4,1,0,8,NULL,8,0,0,0),(3,3,2,6,3,NULL,9,0,0,0),(4,4,2,3,7,NULL,10,0,1,0),(5,3,3,10,NULL,NULL,25,1,0,0),(6,4,3,0,3,NULL,3,0,0,0),(7,3,4,10,NULL,NULL,17,1,0,0),(8,4,4,0,10,NULL,11,0,1,0),(9,3,5,5,2,NULL,7,0,0,0),(10,4,5,1,9,NULL,10,0,1,0),(11,3,6,6,0,NULL,6,0,0,0),(12,4,6,0,0,NULL,0,0,0,1),(13,3,7,1,1,NULL,2,0,0,0),(14,4,7,0,0,NULL,0,0,0,1),(15,3,8,1,0,NULL,1,0,0,0),(16,4,8,3,1,NULL,4,0,0,0),(17,3,9,0,5,NULL,5,0,0,0),(18,4,9,8,2,NULL,14,0,1,0),(19,3,10,10,3,7,20,1,1,0),(20,4,10,4,5,NULL,9,0,0,0);
/*!40000 ALTER TABLE `frame` ENABLE KEYS */;
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
